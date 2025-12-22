import datetime
import json
import os
from pathlib import Path
import pydicom
import logging

from convert_dicomseg2nifti_withref import convert_dicomseg2nifti
from convert_dcms import convert_dicom_series_to_nifti # (in_dir: Path, out_dir: Path)
from convert_dcms import resample_CT_to_PET # (ct_file: Path, pet_file: Path, target_dir: Path)
from convert_dcms import convert_PET_to_SUV # (first_dcm: Path, pet_volume: Path, target_dir: Path):

logger = logging.getLogger(__name__)
DATE_SHIFT = 6247

def dicom_date_to_datetime(dt: str) -> datetime.date:
    """Convert a dicom date string of format YYYYMMDD to python.

    Following: https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html
    """
    dt = dt.strip()

    year = int(dt[:4])
    month = int(dt[4:6])
    day = int(dt[6:8])
    return datetime.date(
        year, month, day
    )


def get_first_file(directory: Path) -> Path:
    """Get the first file in a directory, sorted by name in reverse order."""
    files = sorted(os.listdir(directory))
    if not files:
        raise FileNotFoundError(f"No files found in {directory}")
    return directory / files[0]


def get_injection_time_diffs() -> dict:
    with open("pet_time_since_injection.json") as f:
        return json.load(f)
    

def process_study(study_directory: Path, out_dir: Path, keep_pet: bool = False, keep_ct_orig: bool = False, skip_if_exists: bool = False) -> None:
    """Process a single study directory and convert the DICOM files to NIfTI format."""
    logger.info(f"process {study_directory}")
    # determine pseudonym/timepoint/modality
    patient_dir, study_dir = study_directory.parts[-2:]
    logger.debug(f"extracted {patient_dir=}/{study_dir=}")

    subdirs = {}
    for directory in os.listdir(study_directory):
        if directory == ".DS_Store":
            continue
        example_dcm = pydicom.dcmread(get_first_file(study_directory/directory))
        subdirs[directory] = {
            "modality": example_dcm.Modality,
            "study_date": dicom_date_to_datetime(example_dcm.StudyDate),
            "patient_id": example_dcm.PatientID,
        }

    dates = [v["study_date"] for v in subdirs.values()]
    assert len(set(dates)) == 1
    patient_ids = [v["patient_id"] for v in subdirs.values()]
    assert len(set(patient_ids)) == 1

    injecion_time_diffs = get_injection_time_diffs()

    # map to challenge dataset
    mapped_pseudonym = f"psma_{patient_ids[0].split('_')[1]}"
    mapped_date = f"{dates[0]+datetime.timedelta(days=DATE_SHIFT)}"

    out_ct_orig = out_dir/"imagesTr"/f"{mapped_pseudonym}_{mapped_date}_8888.nii.gz"
    out_ct = out_dir/"imagesTr"/f"{mapped_pseudonym}_{mapped_date}_0000.nii.gz"
    out_pet = out_dir/"imagesTr"/f"{mapped_pseudonym}_{mapped_date}_9999.nii.gz"
    out_suv = out_dir/"imagesTr"/f"{mapped_pseudonym}_{mapped_date}_0001.nii.gz"
    out_seg = out_dir/"labelsTr"/f"{mapped_pseudonym}_{mapped_date}.nii.gz"

    if skip_if_exists and out_ct.exists() and out_suv.exists() and out_seg.exists():
        logger.info(f"Skipping {study_directory} as output files already exist.")
        return

    # determine in_dirs
    for subdir in subdirs:
        if subdirs[subdir]["modality"] == "PT":
            in_dir_pet = study_directory/subdir
        if subdirs[subdir]["modality"] == "CT":
            in_dir_ct = study_directory/subdir
        if subdirs[subdir]["modality"] == "SEG":
            in_dir_seg = study_directory/subdir
    logger.info(f"Matched input dirs to modality:\n - {in_dir_ct} (CT)\n - {in_dir_pet} (PET)\n - {in_dir_seg} (SEG)")

    # convert CT/PET to nifty
    convert_dicom_series_to_nifti(in_dir_ct, out_ct_orig)
    convert_dicom_series_to_nifti(in_dir_pet, out_pet)
    resample_CT_to_PET(out_ct_orig, out_pet, out_ct)
    convert_PET_to_SUV(get_first_file(in_dir_pet), out_pet, out_suv, injecion_time_diffs[mapped_pseudonym+"_"+mapped_date])

    # convert DCM Seg
    dcm_seg = get_first_file(in_dir_seg)
    pet_dicom_list = [in_dir_pet / Path(f) for f in os.listdir(in_dir_pet)]
    convert_dicomseg2nifti(dcm_seg, pet_dicom_list, out_pet, out_seg)

    if not keep_pet:
        out_pet.unlink()
    if not keep_ct_orig:
        out_ct_orig.unlink()

    if not keep_ct_orig and not keep_pet:
        logger.info(f"Created output files:\n - {out_ct} (CT)\n - {out_suv} (PET/SUV)\n - {out_seg} (SEG)")
    elif not keep_pet:
        logger.info(f"Created output files:\n - {out_ct_orig} (CT)\n - {out_ct} (CT)\n  - {out_suv} (PET/SUV)\n - {out_seg} (SEG)")
    elif not keep_ct_orig:
        logger.info(f"Created output files:\n - {out_ct} (CT)\n - {out_pet} (PET/SUV)\n - {out_suv} (PET/SUV)\n - {out_seg} (SEG)")
    else:
        logger.info(f"Created output files:\n - {out_ct_orig} (CT)\n - {out_ct} (CT)\n - {out_pet} (PET/SUV)\n - {out_suv} (PET/SUV)\n - {out_seg} (SEG)")
    return
    

def main(in_dir: Path, out_dir: Path, skip_if_exists: bool = False) -> None:
    """Convert the TCIA DICOM dataset to the format used in the autoPET III challenge."""
    (out_dir/"labelsTr").mkdir(exist_ok=True, parents=True)
    (out_dir/"imagesTr").mkdir(exist_ok=True, parents=True)
    for pseudonym in os.listdir(in_dir):
        if not pseudonym.startswith("PSMA_"):
            continue
        for study_UID in os.listdir(in_dir/pseudonym):
            if study_UID == ".DS_Store":
                continue
            process_study(in_dir/pseudonym/study_UID, out_dir, skip_if_exists=skip_if_exists)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_dir",
        type=Path,
        help="Input directory containing the TCIA DICOM dataset.",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Output directory where the converted dataset will be saved.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversion if output files (CT, SUV, SEG) already exist.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)
    
    main(args.in_dir, args.out_dir, skip_if_exists=args.skip_existing)
