import json
import logging
import hashlib
from pathlib import Path
import tqdm

logger = logging.getLogger(__name__)

def file_hash(file: Path) -> str:
    with open(file, "rb") as f:
        data = f.read()
        m = hashlib.sha256()
        m.update(data)
        m.digest()
    return m.hexdigest()

def main(out_dir: Path, validate_hashes: bool = False ) -> None:
    with open("autopet_v1.1_hashes.json") as f:
        hashes = json.load(f)
    n_files = 0
    n_files_missing = 0
    n_files_ok = 0
    n_files_differ = 0
    for expected_file in tqdm.tqdm(hashes):
        # logger.debug(f"Examining {expected_file}")
        n_files += 1
        filename = out_dir/expected_file
        if not (filename).exists():
            n_files_missing += 1
            continue
            #logger.error(f"File missing: {filename}")
        else:
            if validate_hashes:
                _hash = file_hash(filename)
                if _hash == hashes[expected_file]:
                    n_files_ok += 1
                else:
                    n_files_differ += 1
                    logger.error(f"Hashes differ for {filename}: {_hash} != {hashes[expected_file]}")
            else:
                n_files_ok += 1
    if validate_hashes:           
        logger.info(f"Summary: {n_files} expected, {n_files_missing} missing, {n_files_ok} processed fine, {n_files_differ} differ.")
    else:
        logger.info(f"Summary: {n_files} expected, {n_files_missing} missing, {n_files_ok} processed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "out_dir",
        type=Path,
    )
    parser.add_argument(
        "--validate_hashes",
        action="store_true",
        help="Validate file hashes against autoPET III file hashes. Only possible for non de-faced DICOM input data (v1).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)
    
    main(args.out_dir, validate_hashes=args.validate_hashes)
