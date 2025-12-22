# TCIA PSMA-PET-CT-Lesions DICOM to NIfTI Converter

This repository contains the Docker image and code required to convert the **TCIA PSMA-PET-CT-Lesions** dataset from DICOM format into **NIfTI format**, as used in the [**AutoPET III Grand Challenge**](https://autopet.grand-challenge.org/). The script specifically creates the **PSMA-PET/CT** subset of the challenge dataset and organizes it following the [**nnUNet format**](https://github.com/MIC-DKFZ/nnUNet).

Note: The TCIA PSMA-PET-CT-Lesions dataset currently contains a problematic study: 'PSMA_771c8dc6051db4d7/11-19-1998-NA-PETCT whole-body PSMA-82612'. The DICOM SEG in this study references a non-existent PET SOPInstanceUID. Please remove this study before conversion. This study will be fixed by TCIA soon.


## 📚 Links to Datasets
- **TCIA Dataset:** [10.7937/R7EP-3X37](https://doi.org/10.7937/R7EP-3X37) 
- **NIfTI Dataset v1:** [10.57754/FDAT.5bjzn-0vh28](https://doi.org/10.57754/FDAT.5bjzn-0vh28) 
- **NIfTI Dataset v2:** [10.57754/FDAT.gpeq5-yxy63](https://doi.org/10.57754/FDAT.gpeq5-yxy63) 
- **Challenge Dataset:** [AutoPET III Dataset](https://autopet-iii.grand-challenge.org/dataset/)


## 🧾 Dataset Output Structure

After conversion, the dataset is structured in [nnUnet](https://github.com/MIC-DKFZ/nnUNet) format as follows:

```
nnUNet_dataset/
├── imagesTr/
│   ├── psma_patient1_study1_0000.nii.gz  # CT image (resampled to PET space)
│   ├── psma_patient1_study1_0001.nii.gz  # PET image (SUV units)
│   └── ...
└── labelsTr/
    ├── psma_patient1_study1.nii.gz       # Manual tumor lesion segmentations
    └── ...
```


## 🚀 Usage
The converter is packaged as a Docker image and expects:

- `<input-directory>` mounted at `/in` (TCIA DICOM data)
- `<output-directory>` mounted at `/out` (generated NIfTI dataset)
- Optional behavior is controlled via environment variables


### Option 1: Use Prebuilt Docker Image

1. **Pull Docker Image**

   *(Note: Docker image will be provided soon)*

   ```bash
   docker pull <dockerhub>/tcia_converter_v1
   ```

2. **Run the Docker Container**

   ```bash
   docker run --rm -it \
     -v <input-directory>:/in \
     -v <output-directory>:/out \
     -e CONVERT_FLAGS="--skip-existing" \
     <dockerhub>/tcia_converter_v1
   ```


### Option 2: Build Docker Image Locally

1. **Pull Python Base Image**

   ```bash
   docker pull python:3.8.13
   ```

2. **Build Docker Image**

   Clone this repository and run:

   ```bash
   docker build -t <dockername> .
   ```

3. **Run the Docker Container**

   ```bash
   docker run --rm -it \
     -v <input-directory>:/in \
     -v <output-directory>:/out \
     -e CONVERT_FLAGS="--skip-existing" \
     <dockername>
   ```

### Environment Variables
   - `-e CONVERT_FLAGS="--skip-existing"`: Recommended to skip conversion if output files already exist.
   - `-e VALIDATE_FLAGS="--validate_hashes"`: Currently not recommended for use with TCIA dataset; only use if input DICOM data is original, not additionally de-faced by TCIA.



## 🧩 Additional Notes

* The PET images are converted into **SUV units**.
* The CT images are **resampled** to match PET resolution and spacing.
* Annotations correspond to **manual segmentations of tumor lesions**.
* The resulting dataset is **directly compatible** with nnUNet and the AutoPET III challenge pipeline.
* The resulting dataset only contains the PSMA-PET/CT subset of the AutoPET III challenge dataset. Links for the FDG-PET/CT subset are provided in the [Related Data and Tools](#-related-data-and-tools) section below. 
* If the TCIA DICOM dataset is used as source, the NIfTI files produced by this code will not be identical to the official challenge data, as the TCIA source DICOM files are fully de-faced, while the challenge dataset is not.


## 📦 Related Data and Tools

* **FDG-PET/CT Dataset:** 
  [FDG-PET-CT-Lesions](https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/)

* **FDG Conversion Codebase:**
  [lab-midas/TCIA\_processing](https://github.com/lab-midas/TCIA_processing)


## 📄 License

This repository and associated scripts are released under the [MIT License](https://choosealicense.com/licenses/mit/).


