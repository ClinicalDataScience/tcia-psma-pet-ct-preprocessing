#!/bin/bash
set -e

python convert_dataset.py /in /out ${CONVERT_FLAGS}
python validate_dataset.py /out ${VALIDATE_FLAGS}

