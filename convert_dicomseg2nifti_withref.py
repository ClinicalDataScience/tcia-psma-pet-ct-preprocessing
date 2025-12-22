import logging
import pydicom
import SimpleITK as sitk
from pathlib import Path
import numpy as np
from typing import List

from pydicom.dataset import FileDataset
from pydicom.filereader import dcmread

logger = logging.getLogger(__name__)

def read_dicom_files(dicom_files: List[Path]) -> List[FileDataset]:
    """Read DICOM files and sort them by slice position."""
    datasets = [dcmread(str(dcm)) for dcm in dicom_files]
    return sorted(datasets, key=lambda x: x.ImagePositionPatient[2])

def validate_dicom_orientation(datasets: List[FileDataset]) -> None:
    """Check if the DICOM files have standard LPS orientation."""
    # cf. https://gist.github.com/agirault/60a72bdaea4a2126ecd08912137fe641
    # Check if all orientations of all instances are the same
    orientations = [ds.ImageOrientationPatient for ds in datasets]
    if not all([o == orientations[0] for o in orientations]):
        raise ValueError("DICOM files have different orientations.")
    else:
        orientation = orientations[0]
        orientation = [
            int(i) for i in orientation
        ]  # found '-0' instead of '0' in some cases, cast to int
        if orientation != [1, 0, 0, 0, 1, 0]:
            raise ValueError(
                f"DICOM files are not in LPS orientation, but in {orientation} orientation."
            )
                

def dcm_2d_to_3d_orientation(iop: List[str]) -> np.ndarray:
    """
    Converts a 2D DICOM ImageOrientationPatient (IOP) vector to a 3D orientation matrix in LPS (Left-Posterior-Superior) coordinate system.

    Args:
        iop (List[str]): A list of 6 strings representing the DICOM ImageOrientationPatient attribute, 
                         where the first three values are the direction cosines of the first row (x-direction),
                         and the last three are the direction cosines of the first column (y-direction).

    Returns:
        np.ndarray: A 3x3 orientation matrix (as a flattened tuple) where columns represent the x, y, and z direction vectors in LPS space.

    """
 
    assert len(iop) == 6

    # Extract x-vector and y-vector from image orientation patient
    x_dir = [float(x) for x in iop[:3]]
    y_dir = [float(x) for x in iop[3:]]

    # L2 normalization
    x_dir /= np.linalg.norm(x_dir)  # type: ignore
    y_dir /= np.linalg.norm(y_dir)  # type: ignore

    # Compute perpendicular z-vector
    z_dir = np.cross(x_dir, y_dir)

    orientation = np.stack([x_dir, y_dir, z_dir], axis=1)
    return tuple(orientation.flatten())


def load_dicom_SEG(seg_dicom: str) -> pydicom.Dataset:
    """Load a DICOM SEG file."""
    return pydicom.dcmread(seg_dicom)


def check_dicom_SEG(seg_ds: pydicom.Dataset) -> None:
    """Check if the DICOM SEG file has only one segment and binary segmentation data."""
    if len(seg_ds.SegmentSequence) != 1:
        raise ValueError(f"Only one segment is supported. But found {len(seg_ds.SegmentSequence)} segments.")
    if seg_ds.SegmentationType != "BINARY":
        raise ValueError(f"Only binary segmentation is supported. But found {seg_ds.SegmentationType} for fractional data.")


def get_dicom_SEG_metadata(seg_ds: pydicom.Dataset) -> tuple:
    """
    Extracts relevant metadata from a DICOM SEG (Segmentation) file.

    Args:
        seg_ds (pydicom.Dataset): The DICOM dataset representing the SEG file.

    Returns:
        tuple: A tuple containing:
            - dict_seg_slices (dict): A dictionary mapping Referenced SOP Instance UIDs to their corresponding z-coordinate (ImagePositionPatient[2]).
            - seg_origin (tuple): The (x, y, z) origin of the segmentation, as a tuple of floats.
            - seg_instance_orientation (list): The image orientation (ImageOrientationPatient) as a list of floats.
            - seg_pixel_spacing (list): The pixel spacing (PixelSpacing) as a list of floats.

    """

    seg_instance_orientation = seg_ds.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient
    seg_pixel_spacing = seg_ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing

    dict_seg_slices = {
        frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID: frame.PlanePositionSequence[0].ImagePositionPatient[2]
        for frame in seg_ds.PerFrameFunctionalGroupsSequence
    }

    seg_origin = tuple(float(x) for x in seg_ds.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient)
    return dict_seg_slices, seg_origin, seg_instance_orientation, seg_pixel_spacing

def get_dicom_SEG_pixel_data(seg_ds: pydicom.Dataset) -> np.ndarray:
    """Extract pixel data from the DICOM SEG file."""
    return seg_ds.pixel_array

def get_reference_DICOM_z_positions(ref_dicom_slices: List[pydicom.Dataset]) -> dict:
    """Extract z-positions and SOPInstanceUIDs from reference DICOM slices."""
    return {
        slice.SOPInstanceUID: slice.ImagePositionPatient[2] for slice in ref_dicom_slices
    }

def validate_dicom_SEG_against_DICOM_reference(seg_z_positions: dict, ref_z_positions: dict) -> None:
    """
    Validate that all segmentation SOPInstanceUIDs exist in the reference DICOM and that their corresponding z-positions match within a specified tolerance.

    Args:
        seg_z_positions (dict): A dictionary mapping SOPInstanceUIDs (str) to z-positions (float) for the segmentation slices.
        ref_z_positions (dict): A dictionary mapping SOPInstanceUIDs (str) to z-positions (float) for the reference DICOM slices.

    """
   
    for sopIUID, z_pos in seg_z_positions.items():
        if sopIUID not in ref_z_positions.keys():
            raise ValueError(f"Segmentation slice with SOPInstanceUID {sopIUID} and z-position {z_pos} is not found in the reference DICOM.")
        if not np.allclose(z_pos, ref_z_positions[sopIUID], atol=1e-3): # 1e-3 is the tolerance used for converting from nifti to DICOM in convert_nifti2dicomseg_highdicom.py
            raise ValueError(f"Z-position {z_pos} of segmentation slice with SOPInstanceUID {sopIUID} does not match reference DICOM z-position {ref_z_positions[sopIUID]}.")
    else: 
        logger.info("All SOPInstanceUIDs of the segmentation are found in the reference DICOM and z-positions match.")

def find_index_with_tolerance(value: float, array: List[float], atol: float = 1e-5) -> int: 
    """
    Find the index of a value in an array with a specified tolerance.

    Args:
        value (float): The value to find.
        array (list of float): The array to search in.
        tolerance (float): The allowable tolerance for matching.

    Returns:
        int: The index of the closest matching value.

    """
    for i, elem in enumerate(array):
        if abs(value - elem) <= atol:
            return i
    raise ValueError(f"No value found within tolerance {atol} for {value} in {array}.")

def get_uniform_z_spacing(z_positions: List[float], tolerance: float = 1e-3) -> float:
    """
    Calculate the uniform z-spacing from a list of z-positions.
    
    Args:
        z_positions (list of float): The z-positions to calculate spacing from.
        tolerance (float): The allowable tolerance for spacing differences.

    Returns:
        float: The uniform z-spacing.

    """
    # Calculate differences between consecutive z-positions
    z_spacings = np.diff(sorted(z_positions))
    
    # Check if all spacings are uniform within the given tolerance
    if np.all(np.abs(z_spacings - z_spacings[0]) <= tolerance):
        return z_spacings[0]
    else:
        raise ValueError(
            f"Z-spacings are not uniform within tolerance {tolerance}. "
            f"Spacings: {z_spacings}"
        )


def map_dicom_SEG_data_to_reference_volume(seg_pixel_array: np.ndarray, seg_z_positions: List[float], ref_z_positions: List[float], dtype: np.dtype) -> tuple:
    """
    Maps a segmentation array to align with the reference DICOM slices based on z-positions.

    Args:
        seg_pixel_array (np.ndarray): 3D numpy array containing segmentation data, with shape 
            (num_seg_slices, height, width). Each slice corresponds to a z-position in seg_z_positions.
        seg_z_positions (list of float): List or array of z-positions for each segmentation slice.
        ref_z_positions (list of float): List or array of z-positions for the reference DICOM volume.
        dtype (np.dtype): Data type for the output segmentation array.

    Returns:
        tuple:
            - new_seg_array (np.ndarray): 3D numpy array of shape (num_ref_slices, height, width), 
              where segmentation slices are mapped to the corresponding reference DICOM slices. 
              Slices without segmentation data are filled with zeros.
            - new_z_spacing (float): The uniform spacing between z-positions in the reference subvolume.

    """

    min_z, max_z = min(seg_z_positions), max(seg_z_positions)
    ref_z_positions_sub = [z for z in ref_z_positions if min_z <= z <= max_z]
    new_z_spacing = get_uniform_z_spacing(ref_z_positions_sub)

    new_seg_array_shape = (len(ref_z_positions_sub),) + seg_pixel_array.shape[1:] # Note: dicom SEGs created with highdicom contain full slices, thus the shape of the array is the same as in the reference DICOM
    new_seg_array = np.zeros(new_seg_array_shape, dtype=dtype)

    for seg_index, z_pos in enumerate(seg_z_positions):
        try:
            ref_index = find_index_with_tolerance(z_pos, ref_z_positions_sub, atol = 1e-3)
        except:
            raise ValueError(f"Z-position {z_pos} of segmentation slice not found in reference DICOM slices.")
        new_seg_array[ref_index, :, :] = seg_pixel_array[seg_index, :, :]

    return new_seg_array, new_z_spacing


def create_seg_nifti(seg_array: np.ndarray, seg_origin: tuple, seg_direction: np.ndarray, seg_spacing: tuple) -> sitk.Image: 
    """
    Creates a SimpleITK NIfTI image from a segmentation array with specified origin, direction, and spacing.

    Args:
        seg_array (np.ndarray): The segmentation array to convert into a NIfTI image.
        seg_origin (tuple): The physical origin of the image (x, y, z).
        seg_direction (np.ndarray): The direction cosine matrix as a flat array or matrix.
        seg_spacing (tuple): The voxel spacing along each dimension (x, y, z).

    Returns:
        sitk.Image: The resulting SimpleITK image with the specified metadata.

    """

    seg_image = sitk.GetImageFromArray(seg_array)
    seg_image.SetDirection(seg_direction)
    seg_image.SetOrigin(seg_origin)
    seg_image.SetSpacing(seg_spacing)
    return seg_image


def resample_segmentation_to_reference(seg_image: sitk.Image, ref_nifti: sitk.Image) -> sitk.Image:
    """
    Resample a segmentation SimpleITK image to match the geometry of a reference NIfTI image.

    Args:
        seg_image (sitk.Image): The segmentation image to be resampled.
        ref_nifti (sitk.Image): The reference NIfTI image whose geometry (spacing, origin, direction, and size) will be matched.

    Returns:
        sitk.Image: The resampled segmentation image aligned to the reference image geometry.

    """

    try:
        return sitk.Resample(seg_image, ref_nifti, interpolator=sitk.sitkNearestNeighbor)
    except Exception as e:
        raise RuntimeError(f"Error resampling segmentation: {e}")
    

def validate_nifti_metadata_alignment(seg_nifti: sitk.Image, ref_nifti: sitk.Image, tolerance: float = 1e-3) -> None:
    """
    Checks if the spacing and direction of a segmentation NIfTI image 
    matches the reference NIfTI image within a specified tolerance.
    
   Args:
        seg_nifti (SimpleITK.Image): The segmentation NIfTI image to check.
        ref_nifti (SimpleITK.Image): The reference NIfTI file.
        tolerance (float): Tolerance for comparison (default is 1e-3).

    """
    
    # Check spacing with tolerance
    seg_spacing = np.array(seg_nifti.GetSpacing())
    ref_spacing = np.array(ref_nifti.GetSpacing())
    if not np.allclose(seg_spacing, ref_spacing, atol=tolerance):
        raise AssertionError(
            f"Spacing mismatch: seg_nifti spacing: {seg_spacing}, "
            f"ref_nifti spacing: {ref_spacing}"
        )
    
    # Check direction with tolerance
    seg_direction = np.array(seg_nifti.GetDirection())
    ref_direction = np.array(ref_nifti.GetDirection())
    if not np.allclose(seg_direction, ref_direction, atol=tolerance):
        raise AssertionError(
            f"Direction mismatch: seg_nifti direction: {seg_direction}, "
            f"ref_nifti direction: {ref_direction}"
        )

def convert_dicomseg2nifti(seg_dicom: str, ref_dicom: List[Path], ref_nifti: str, output_path: str) -> None:
    """
    Convert a DICOM SEG file to a NIfTI file using the reference DICOM and its conversion to NIfTI.

    Parameters:
        seg_dicom (str): Path to the DICOM SEG file.
        ref_dicom (str): Path to the reference DICOM image directory.
        ref_nifti (str): Path to the reference NIfTI file.
        output_path (str): Path to save the output NIfTI file.

    Note: 
    This custom function is needed since standard libraries don't work for this dataset, which requires conversion of inconsistent slice increments
    - dicom_seg and dcm2niix assume slice spacing = slice thickness even if slice thickness is unequal to slice spacing
    - dicom2nifti throws errors like 
        - "Conversion of inconsistent slice increment with resampling not supported for multiframe"
        - "TypeError: Data type not understood by NumPy: format='uint1', PixelRepresentation=0, BitsAllocated=1"

    """
    # Load segmentation DICOM
    seg_ds = load_dicom_SEG(seg_dicom)
    check_dicom_SEG(seg_ds)
    dict_seg_z_positions, seg_origin, seg_instance_orientation, seg_pixel_spacing = get_dicom_SEG_metadata(seg_ds)
    seg_pixel_array = get_dicom_SEG_pixel_data(seg_ds) # stacked 2D arrays of slices containing segmentations, full 3D array for empty masks

    # Load and validate reference DICOM
    list_ref_ds = read_dicom_files(ref_dicom)
    validate_dicom_orientation(list_ref_ds)
    dict_ref_z_positions = get_reference_DICOM_z_positions(list_ref_ds)

    # Validate matching of SOPInstanceUIDs and corresponding z-positions 
    validate_dicom_SEG_against_DICOM_reference(dict_seg_z_positions, dict_ref_z_positions)

    # Map (sparse) segmentation to reference DICOM slices to create seg volume
    seg_array, seg_z_spacing = map_dicom_SEG_data_to_reference_volume( 
        seg_pixel_array, list(dict_seg_z_positions.values()), list(dict_ref_z_positions.values()), dtype=seg_pixel_array.dtype
    )
 
    # Create NIfTI image from segmentation array
    seg_direction = dcm_2d_to_3d_orientation(seg_instance_orientation)
    seg_spacing = (seg_pixel_spacing[0], seg_pixel_spacing[1], seg_z_spacing)
    seg_nifti = create_seg_nifti(seg_array, seg_origin, seg_direction, seg_spacing)

    # Check if seg_nifti and ref_nifti have the same direction and spacing
    ref_nifti = sitk.ReadImage(ref_nifti)
    reoriented_ref_nifti = sitk.DICOMOrient(ref_nifti, 'LPS')
    validate_nifti_metadata_alignment(seg_nifti, reoriented_ref_nifti, tolerance=1e-3)
  
    # Resample segmentation to match reference NIfTI geometry
    resampled_seg_image = resample_segmentation_to_reference(seg_nifti, ref_nifti)

    # Save the resampled segmentation
    sitk.WriteImage(resampled_seg_image, output_path)
