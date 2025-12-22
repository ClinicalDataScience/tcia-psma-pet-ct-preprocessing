import pydicom
import dicom2nifti
import dicom2nifti.settings as settings
from pathlib import Path
import nibabel as nib
import nilearn.image
import logging
import numpy as np

# Disable slice increment validation due to bug in dicom2nifti 2.4.8
settings.disable_validate_slice_increment()
# Enable resampling to handle inconsistent slice increments properly
settings.enable_resampling()
settings.set_resample_spline_interpolation_order(1)
settings.set_resample_padding(-1000)

logger = logging.getLogger(__name__)

class StudyConverterError(Exception):
    """Raised when something goes wrong in the Study Converter."""

    pass


def convert_dicom_series_to_nifti(in_dir: Path, target_file: Path) -> None:
    """Converts a DICOM series in a given directory to either "CT.nii.gz" or "PET.nii.gz", depending on dicomfile.Modality

    Parameters:
        in_dir (Path): Directory containing a single DICOM series.
        target_file (Path): Path to the output NIfTI file (e.g., "CT.nii.gz" or "PET.nii.gz").

    """
    
    # convert to nifti
    dicom2nifti.convert_dicom.dicom_series_to_nifti(
        in_dir, output_file=target_file, reorient_nifti=True
    )


def _determine_SUV_factor(first_dcm: Path, time_diff: float) -> float:
    """
    Calculate the Standardized Uptake Value (SUV) correction factor for a PET DICOM file.

    Parameters:
        first_dcm (Path): Path to PET DICOM file used to extract metadata.
        time_diff (float): Time difference in seconds between radiotracer injection and image acquisition.

    Returns:
        float: The calculated SUV correction factor.

    """

    ds_pet = pydicom.read_file(first_dcm)
    # logger.debug(f"PET: {first_dcm=}")

    pet_total_dose = ds_pet.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    pet_time_since_injection_seconds = time_diff
    pet_half_life = ds_pet.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    if round(pet_half_life / 60) not in [110, 68]:
        raise StudyConverterError(
            f"RadionuclideHalfLife is not 110 (18F) or 68 (68Ga) minutes for patient {first_dcm.PatientID} at timepoint {first_dcm.StudyDate}. Wrong radionuclide."
        )


    weight = ds_pet.PatientWeight
    pet_total_dose = ds_pet.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose

    act_dose = pet_total_dose * 0.5 ** (
        pet_time_since_injection_seconds / pet_half_life
    )
    suv_factor = 1000 * weight / act_dose
    logger.debug(f"SUV factor components: {weight=}, {pet_total_dose=}, {pet_time_since_injection_seconds=}")
    logger.debug(f"SUV factor components (cont.): {pet_half_life=}, {act_dose=}")
    logger.debug(f"SUV factor to be applied: {suv_factor}")
    return suv_factor


def convert_PET_to_SUV(first_dcm: Path, pet_volume: Path, target_file: Path, time_diff: float) -> nib.Nifti1Image:
    """Converts a PET image to an SUV (Standardized Uptake Value) image and saves the result as a NIfTI volume.

    Parameters:
        first_dcm (Path): Path to a DICOM file used to extract metadata for determining the SUV conversion factor.
        pet_volume (Path): Path to the input NifTI PET volume to be converted.
        target_file (Path): Path where the converted SUV NifTI volume will be saved.
        time_diff (float): Time difference between radiotracer administration and image acquisition.

    Returns:
        nib.Nifti1Image: The converted SUV NifTI image object.

    """
  
    suv_factor = _determine_SUV_factor(first_dcm, time_diff)

    PET_nib = nib.load(pet_volume)

    affine = PET_nib.affine
    pet_data = PET_nib.get_fdata()
    PET_suv_data = (pet_data * suv_factor).astype(np.float32)
    suv_nib = nib.Nifti1Image(PET_suv_data, affine)

    nib.save(suv_nib, target_file)

    return suv_nib


def resample_CT_to_PET(ct_file: Path, pet_file: Path, CT_resampled_filename: Path) -> None:
    """Resamples a NIfTI CT volume to match the spatial resolution and shape of a reference NIfTI PET volume.

    Parameters:
        ct_file (Path): Path to the input NIfTI CT volume file.
        pet_file (Path): Path to the reference NIfTI PET volume file.
        CT_resampled_filename (Path): Path where the resampled NIfTI CT volume will be saved.

    """

    CT_nib = nib.load(ct_file)
    pet_nib = nib.load(pet_file)

    CT_resampled_nib = nilearn.image.resample_to_img(
        CT_nib, pet_nib, fill_value=-1024
    )

    nib.save(CT_resampled_nib, CT_resampled_filename)