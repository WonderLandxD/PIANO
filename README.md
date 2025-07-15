# 🎶 PIANO (Preview): Pathology Image ANalysis Orchestrator 

**PIANO** is a simple PyTorch library for pathology image analysis. It helps you generate patches from whole-slide images (WSIs), use pathology foundation models for feature extraction, fine-tuning patch-level and slide-level tasks, and more! 🚀

**Why PIANO ? ✨**

•   **Diverse WSI pre-processing:** a step-by-step pipeline to make WSI pre-processing fully transparent, supporting the entire flow: 

`WSI → patches (optional) → Multi-D patch-level features (via patch foundation models) → 1-D slide-level feature (via slide foundation models)`

•	**Convenient model function interface:** call several types of conputational pathology model with a single line of code, including `patch-level foundation models`, `slide-level foundation models`, `multi-instance learning models (MIL)`.

•	**Diverse scripts:** clear, transparent fine-tuning or evaluation code for `ROI classfication`, `WSI classification (MIL and slide foundation models)`, and `WSI survival analysis (MIL)`.

•	🌍 More functionality will be released soon. Pull requests with additional scripts are warmly encouraged.

---------

<details>
<summary>📰 Click to view News</summary>

**2025-07-15:** Modify the model interface, add MIL models (`abmil`, `gated_abmil`, `clam_sb`, `clam_mb`, `transmil`, `dgrmil`, `dtfdmil`, `dsmil`, `ilramil`, `wikg`, `s4mil`, `amdmil`, `mean_pool`, `max_pool`, `2dmamba`, `m4`). Add slide foundation models (FMs) (`chief`, `gigapath`, `titan`, `prism`, `madeleine`). Add patch FMs (`pathorchestra`). Add patch-level classification fine-tuning codes. Add slide-level classification fine-tuning codes. Add survival analysis fine-tuning codes. Add slide-level KNN and ProtoNet codes. Modify the README.md file.

**2025-05-20:** Added fine-tuning codes for slide-level classification tasks. Optimized some codes and README.md. 

**2025-03-19:** Added fine-tuning codes for patch-level classification tasks. Optimized some codes. 

**2025-03-18:** Enhanced model flexibility with local loading support, added `CTransPath (CHIEF-based weights)` model, and introduced YAML-based preprocessing customization and coordinate saving.
</details>

## 🎈 Installation

First, clone the repo and cd into the directory:

```
git clone https://github.com/WonderLandxD/PIANO.git
cd PIANO
```

Next, create a conda env and install the library in editable mode. We directly use the newly created conda environment from the [opensdpc](https://github.com/WonderLandxD/opensdpc/tree/main) library:
```
conda activate piano
pip install -e .
```

## 📚 Usage

### *Define a pretrained pathology foundation model*
We use [PLIP](https://www.nature.com/articles/s41591-023-02504-3) as an example.

***Set up Hugging Face***

```bash
# Change the default path of HuggingFace cache directory
mkdir YOUR_PATH_TO_SAVE_HUGGINGFACE_CACHE

# Open the ~/.bashrc file
vim ~/.bashrc

# Add the following command to ~/.bashrc in the last line
export HF_HOME='YOUR_PATH_TO_SAVE_HUGGINGFACE_CACHE'

# Apply the changes
source ~/.bashrc
```

You can see if the HuggingFace cache directory is set up correctly by running the following command:
```bash
env | grep HF_HOME
```

***Download the pretrained weights***
```bash

# (Optional) Add the HuggingFace mirror if you encounter issues downloading models
# export HF_ENDPOINT='https://hf-mirror.com'

# Some models are accessible after you accept the conditions. You need first to click the "Agree and access repository" button and then create a Hugging Face write token to download the weights.

# Log in to Hugging Face (replace with your write token)
huggingface-cli login --token HUGGINGFACE_WRITE_TOKEN

# Set the model name from huggingface and download the model
export MODEL_NAME='vinid/plip'
huggingface-cli download $MODEL_NAME
```

***Review all the model name and their path***
```bash
# Install the huggingface_hub if you haven't done so
# pip install huggingface_hub

# Scan the cache directory
huggingface-cli scan-cache

# (Optional) If you want to delete the cache directory, you can run the following command:
huggingface-cli delete-cache
```

***Define a Patch FM in Python (PLIP model as an example)***

```python
# Load the PLIP model from Hugging Face
from piano.model.patch_encoder import create_patch_encoder
model = create_patch_encoder(model_name="plip").cuda()

# If you want to load the model from a local path, you can do the following, but note that not all models support the local loading.

# model = piano.create_model(model_name="plip", local_dir=True,  checkpoint_path="YOUR_LOCAL_PATH_TO_MODEL_CHECKPOINT").cuda()
```

***Define a Slide FM in Python (CHIEF model as an example)***

```python
# Load the CHIEF model from Hugging Face
from piano.model.slide_encoder import create_slide_encoder
model = create_slide_encoder(model_name="chief").cuda()
```

***Define a MIL model in Python (ABMIL model as an example)***

```python
# Load the ABMIL model from Hugging Face
from piano.model.mil_encoder import create_mil_encoder
model = create_mil_encoder(model_name="abmil").cuda()
```

***More codes for model definition***

See [./scripts/tutorial.ipynb](scripts/tutorial.ipynb) to see more details about patch FM, slide FM, MIL model loading, and using the model to create embeddings.


### **Patch Foundation Model Checkpoint Paths and Sources**
| **Model Name**           | **Output Dimension** | **Model Checkpoint Path**          | **Model Weight Link** | **Paper/Codebase/Website Link** |
|--------------------------|----------------------|------------------------------------|-----------------------------------------------------------------------------------|--------------|
| `plip`                   | 512                  | `vinid/plip`                       | [Hugging Face - vinid/plip](https://huggingface.co/vinid/plip)                    | [A visual–language foundation model for pathology image analysis using medical Twitter](https://www.nature.com/articles/s41591-023-02504-3) |
| `openai_clip_p16`        | 512                  | `openai/clip-vit-base-patch16`     | [Hugging Face - openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16) | [Learning transferable visual models from natural language supervision](https://proceedings.mlr.press/v139/radford21a) |
| `conch_v1`*               | 512                  | `hf_hub:MahmoodLab/conch`          | [Hugging Face - MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH) | [A visual-language foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02856-4) |
| `uni_v1`                 | 1024                 | `hf-hub:MahmoodLab/uni`            | [Hugging Face - MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) | [Towards a general-purpose foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02857-3) |
| `uni_v2`                 | 1536                 | `hf-hub:MahmoodLab/UNI2-h`         | [Hugging Face - MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) | [Github Page from *github.com/mahmoodlab/UNI*](https://github.com/mahmoodlab/UNI) |
| `prov_gigapath`          | 1536                 | `hf_hub:prov-gigapath/prov-gigapath` | [Hugging Face - prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) | [A whole-slide foundation model for digital pathology from real-world data](https://www.nature.com/articles/s41586-024-07441-w) |
| `virchow_v1`             | 2560                 | `hf-hub:paige-ai/Virchow`          | [Hugging Face - paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow) | [A foundation model for clinical-grade computational pathology and rare cancers detection](https://www.nature.com/articles/s41591-024-03141-0) |
| `virchow_v2`             | 2560                 | `hf-hub:paige-ai/Virchow2`         | [Hugging Face - paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) | [Virchow2: scaling self-supervised mixed magnification models in pathology](https://arxiv.org/pdf/2408.00738) |
| `musk`                   | 2048                 | `hf_hub:xiangjx/musk`              | [Hugging Face - xiangjx/musk](https://huggingface.co/xiangjx/musk) | [A vision–language foundation model for precision oncology](https://www.nature.com/articles/s41586-024-08378-w) |
| `h_optimus_0`            | 1536                 | `hf-hub:bioptimus/H-optimus-0`     | [Hugging Face - bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) | [H-Optimus-0: An open-source foundation model for histology.](https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0) |
| `h_optimus_1`            | 1536                 | `hf-hub:bioptimus/H-optimus-1`     | [Hugging Face - bioptimus/H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) | [H-Optimus-1: The leading foundation model for histology](https://www.bioptimus.com/h-optimus-1#section1) |
| `phikon_v2`              | 768                  | `owkin/phikon-v2`                  | [Hugging Face - owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2) | [Phikon-v2, a large and public feature extractor for biomarker prediction](https://arxiv.org/abs/2409.09173) |
| `ctranspath`            | 768                  | `JWonderLand/CHIEF_unofficial`                  | [Github - Xiyue-Wang/TransPath](https://github.com/Xiyue-Wang/TransPath) | [Transformer-based unsupervised contrastive learning for histopathological image classification](https://www.sciencedirect.com/science/article/abs/pii/S1361841522002043) |
| `pathorchestra`         | 1024                 |  `AI4Pathology/PathOrchestra`            | [Hugging Face - AI4Pathology/PathOrchestra](https://huggingface.co/AI4Pathology/PathOrchestra) | [PathOrchestra: A Comprehensive Foundation Model for Computational Pathology with Over 100 Diverse Clinical-Grade Tasks](https://arxiv.org/pdf/2503.24345)

None: Models marked with * require installation of their original codebase's local libraries to work properly.

### **Slide Foundation Model Checkpoint Paths and Sources**
| **Model Name**           | **Output Dimension** | **Model Checkpoint Path**          | **Model Weight Link** | **Paper/Codebase/Website Link** |
|--------------------------|----------------------|------------------------------------|-----------------------------------------------------------------------------------|--------------|
| `chief`                  | 768                 | `JWonderLand/CHIEF_unofficial` | [Hugging Face - JWonderLand/CHIEF_unofficial](https://huggingface.co/JWonderLand/CHIEF_unofficial) | [A Pathology Foundation Model for Cancer Diagnosis and Prognosis Prediction](https://www.nature.com/articles/s41586-024-07894-z) |
| `gigapath`               | 768                 | `hf_hub:prov-gigapath/prov-gigapath` | [Hugging Face - prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) | [A whole-slide foundation model for digital pathology from real-world data](https://www.nature.com/articles/s41586-024-07441-w) |
| `titan`                  | 768                 | `MahmoodLab/TITAN` | [Hugging Face - MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN) | [Multimodal Whole Slide Foundation Model for Pathology](https://arxiv.org/abs/2411.19666) |
| `prism`                  | 1536                 | `paige-ai/Prism` | [Hugging Face - paige-ai/Prism](https://huggingface.co/paige-ai/Prism) | [PRISM: A Multi-Modal Generative Foundation Model for Slide-Level Histopathology](https://doi.org/10.48550/arXiv.2405.10254) |
| `madeleine`              | 1536                 | `MahmoodLab/madeleine` | [Hugging Face - MahmoodLab/madeleine](https://huggingface.co/MahmoodLab/madeleine) | [Multistain Pretraining for Slide Representation Learning in Pathology](https://arxiv.org/abs/2408.02859) |


### **MIL Model Checkpoint Paths and Sources**
| **Model Name**            | **Paper/Codebase/Website Link** |
|----------------------------|----------------------------|
| `mean_pool`                | [-](-) |
| `max_pool`                 | [-](-) |
| `abmil`                    | [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712) |
| `gated_abmil`              | [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712) |
| `clam_sb`                  | [Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images](https://www.nature.com/articles/s41551-020-00682-w) |
| `clam_mb`                  | [Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images](https://www.nature.com/articles/s41551-020-00682-w) |	
| `transmil`                 | [TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification](https://arxiv.org/abs/2106.00908) |
| `dgrmil`                   | [DGR-MIL: Exploring Diverse Global Representation in Multiple Instance Learning for Whole Slide Image Classification](https://arxiv.org/abs/2407.03575) |
| `dtfdmil`                  | [DTFD-MIL: Double-Tier Feature Distillation Multiple Instance Learning for Histopathology Whole Slide Image Classification](https://arxiv.org/abs/2203.12081) |
| `dsmil`                    | [Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning](https://arxiv.org/abs/2011.08939) |
| `ilramil`                  | [Exploring Low-Rank Property in Multiple Instance Learning for Whole Slide Image Classification](https://openreview.net/forum?id=01KmhBsEPFO) |
| `wikg`                     | [Dynamic Graph Representation with Knowledge-aware Attention for Histopathology Whole Slide Image Analysis](https://arxiv.org/abs/2403.07719) |
| `s4mil`                    | [Structured State Space Models for Multiple Instance Learning in Digital Pathology](https://arxiv.org/abs/2306.15789) |
| `amdmil`                   | [Agent Aggregator with Mask Denoise Mechanism for Histopathology Whole Slide Image Analysis](https://arxiv.org/abs/2409.11664) |
| `2dmamba`                  | [2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification](https://arxiv.org/abs/2412.00678) |
| `m4`                       | [M4: Multi-proxy multi-gate mixture of experts network for multiple instance learning in histopathology image analysis](https://www.sciencedirect.com/science/article/pii/S1361841525001082) |


## 📚 Scripts

### WSI Preprocessing

### *Want to create patch features from a csv file containing the list of WSIs directly?*

```bash
cd scripts/wsi_preprocess

# Generate a CSV file containing the list of WSIs
python 1_run_generate_wsi_list.py --data_folder ROOT_DIRECTORY_PATH_CONTAINING_WSI_FILES --dataset_name DATASET_NAME --save_dir DIRECTORY_TO_SAVE_CSV_FILE

# Extract patch-level features from a csv file containing the list of WSIs, see more argparse parameters in the script
python onestep_patch_features.py --output_dir OUTPUT_DIRECTORY_PATH --csv_path CSV_FILE_PATH --model_name PATCH_FOUNDATION_MODEL_NAME --gpu_id GPU_ID --batch_size BATCH_SIZE --patch_size PATCH_SIZE --overlap OVERLAP --wsi_level WSI_LEVEL --blank_TH BLANK_THRESHOLD --kernel_size KERNEL_SIZE
```

### *Creating patches from WSIs*

```bash
cd scripts/wsi_preprocess

# Generate a CSV file containing the list of WSIs
python 1_run_generate_wsi_list.py --data_folder ROOT_DIRECTORY_PATH_CONTAINING_WSI_FILES --dataset_name DATASET_NAME --save_dir ../WSI_DATA/wsi_list_csv

# Generate patches from all WSIs in the CSV file, see more argparse parameters in the script
python 2_run_generate_patches.py --n_thread 8 --csv_path PATH_TO_CSV_FILE --save_dir DIRECTORY_TO_SAVE_PATCHES
```

**Example from the CPTAC Lung cohort directory:**
```
DIRECTORY_TO_SAVE_PATCHES/
	CPTAC-LUAD_2025-05-19/
		├── C3L-04365-28
    			├── no000000_000003072x_000006144y.jpg
    			├── no000001_000003072x_000009216y.jpg
    			├── no000002_000006144x_000003072y.jpg
    			├── no000003_000006144x_000006144y.jpg
    			├── no000004_000006144x_000009216y.jpg
    			├── no000005_000009216x_000003072y.jpg
    			├── no000006_000009216x_000006144y.jpg
    			├── no000007_000009216x_000009216y.jpg
    			├── no000008_000012288x_000003072y.jpg
    			├── no000009_000012288x_000006144y.jpg
    			├── no000010_000015360x_000006144y.jpg
    			├── no000011_000018432x_000006144y.jpg
    			└── thumbnail/
                      └── x20_thumbnail.jpg
        └── ...
```

### *Creating patch features for WSIs (from patch directories)*

```bash
cd scripts/wsi_preprocess

# Generate a CSV file containing the list of patch directories
python 3_run_create_patchdir_list.py --data_folder ROOT_DIRECTORY_PATH_CONTAINING_PATCHES_FILES --dataset_name DATASET_NAME --save_dir DIRECTORY_TO_SAVE_PATCH_DIR_LIST_CSV_FILE

# Extract patch-level features from a patch directory or a csv file containing the list of patch directories, see more argparse parameters in the script
python 4_run_create_wsi_features.py --batch_size BATCH_SIZE --model_name PATCH_FOUNDATION_MODEL_NAME --gpu_ids GPU_ID_1 GPU_ID_2 ... --num_processes NUM_PROCESSES --save_dir save_directory --csv_path CSV_FILE_PATH --amp AMP_TYPE
```

### *Creating slide features for WSIs (from patch features)*

```bash
cd scripts/wsi_preprocess

# Generate a CSV file containing the list of patch features
python 5_run_create_slide_features.py --data_folder ROOT_DIRECTORY_PATH_CONTAINING_PATCH_FEATURES_FILES --dataset_name DATASET_NAME --save_dir DIRECTORY_TO_SAVE_SLIDE_FEATURES_CSV_FILE

# Extract slide-level features from a csv file containing the list of patch features, see more argparse parameters in the script
python 6_run_create_one_features.py --batch_size BATCH_SIZE --model_name SLIDE_FOUNDATION_MODEL_NAME --gpu_ids GPU_ID_1 GPU_ID_2 ... --num_processes NUM_PROCESSES --save_dir save_directory --csv_path CSV_FILE_PATH --amp AMP_TYPE
```

---

### ROI-level Task Evaluation

### 🪐 ROI/Patch-level image classification using Linear-probing

```bash
cd scripts/roi_classification

python run_roi_train.py --data_json YOUR_DATA_JSON --transform_config ./transform_configs/roi_classification_transforms.yaml --model_name PATCH_MODEL_NAME --training_mode linear_probe --save_metric bal_accuracy --save_dir YOUR_SAVE_DIR

python run_roi_infer.py --test_json YOUR_DATA_JSON --transform_config ./transform_configs/roi_classification_transforms.yaml --model_name PATCH_MODEL_NAME --training_mode linear_probe --finetune_ckpt YOUR_SAVE_PTH_FILE --output YOUR_SAVE_DIR
```

---

### WSI-level Task Evaluation

### 🛰️ Slide-lvel WSI classification using MIL

```bash
cd scripts/wsi_classification

# k-fold train-valid 
python run_mil_kfold.py --data_json YOUR_DATA_JSON --pfm_name PATCH_MODEL_NAME --mil_name MIL_MODEL_NAME --save_dir YOUR_SAVE_DIR

# train-valid-test with k-seed
python run_mil_tvt_kseed.py --data_json YOUR_DATA_JSON --pfm_name PATCH_MODEL_NAME --mil_name MIL_MODEL_NAME --start_seed 30 --end_seed 35 --save_dir YOUR_SAVE_DIR
```

### 🌛 Slide-level WSI survival using MIL

```bash
cd scripts/wsi_survival

# k-fold trian-valid
python run_mil_surv_kfold.py --data_json YOUR_DATA_JSON --pfm_name PATCH_MODEL_NAME --mil_name MIL_MODEL_NAME --save_dir YOUR_SAVE_DIR
```

### 🌞 Slide-level WSI classification with K-NN and ProtoNet

```bash
cd scripts/wsi_classification

# k-fold train-test with 20-NN and 20-ProtoNet
python run_sfm_knn_proto_kfold.py --data_json YOUR_DATA_JSON --pfm_name SLIDE_MODEL_NAME --k 20 --output_dir None

# train-test with 20-NN and 20-ProtoNet using 1000 bootstrap
python run_sfm_knn_proto_bootstrap.py --data_json YOUR_DATA_JSON --pfm_name SLIDE_MODEL_NAME --k 20 --n_bootstrap 1000 --output_dir None
```

For WSI-level tasks, the json file should be in the following format (`"patch_dir"` is optional):

```json
{
	"train": [
		{
			"feat_path": "slide_1.pth",
			"label": "label_1",
			"patch_dir": "PATH_TO_PATCH_DIRECTORY/slide_1"
		},
		{
			"feat_path": "slide_2.pth",
			"label": "label_2",
			"patch_dir": "PATH_TO_PATCH_DIRECTORY/slide_2"
		},
		{
			"feat_path": "slide_3.pth",
			"label": "label_3",
			"patch_dir": "PATH_TO_PATCH_DIRECTORY/slide_3"
		}
		...
	],
	"valid": [
		{
			"feat_path": "slide_4.pth",
			"label": "label_4",
			"patch_dir": "PATH_TO_PATCH_DIRECTORY/slide_4"
		}
		...
	],
	"test": [
		{
			"feat_path": "slide_5.pth",
			"label": "label_5",
			"patch_dir": "PATH_TO_PATCH_DIRECTORY/slide_5"
		}
		...
	]
}
```











### 🪐 *Generating patches and extracting patch-level features from histopathology WSIs.*

---

***STEP 1 - Generate a CSV file containing the list of WSIs.***

We first run the command `scripts/wsi_preprocess/1_run_generate_wsi_list.py` as follows:

```bash
cd scripts/wsi_preprocess

python 1_run_generate_wsi_list.py --data_folder ROOT_DIRECTORY_PATH_CONTAINING_WSI_FILES --dataset_name DATASET_NAME --save_dir ../WSI_DATA/wsi_list_csv
```

> **--data_folder ROOT_DIRECTORY_PATH_CONTAINING_WSI_FILES** is the path where the script will recursively search for all WSI files with extensions '.svs', '.sdpc', '.tiff', '.tif', '.ndpi'. The script will find all matching files in this directory and its subdirectories. 

> **--dataset_name DATASET_NAME** is your custom name for the dataset, which will be used in the output CSV filename.

> **--save_dir YOUR_DIRECTORY_TO_SAVE_CSV_FILE** is the directory where the CSV file containing the list of WSI files will be saved. (Default: `../WSI_DATA/wsi_list_csv`)

 ```bash
# You can customize the file types to search for using the `--additional_file_types` parameter. For example, to include `.mrxs` files in addition to the default extensions:

cd scripts/wsi_preprocess

python 1_run_generate_wsi_list.py --data_folder ROOT_DIRECTORY_PATH_CONTAINING_WSI_FILES --dataset_name DATASET_NAME --save_dir DIRECTORY_TO_SAVE_CSV_FILE --additional_file_types .mrxs
 ```

---
***STEP 2 - Generate patches from all WSIs in the CSV file.***

Next, run the command `scripts/wsi_preprocess/2_run_generate_patches.py` as follows. We recommend using **--n_thread 8** (8 processes) since it works on regular CPU:

```bash
cd scripts/wsi_preprocess

python 2_run_generate_patches.py --n_thread 8 --csv_path PATH_TO_CSV_FILE --save_dir DIRECTORY_TO_SAVE_PATCHES
```

> **--csv_path PATH_TO_CSV_FILE** is the path to the CSV file containing the list of WSI files. (File from step 1)

> **--save_dir DIRECTORY_TO_SAVE_PATCHES** is the directory where the patches will be saved. The script will create a subdirectory with the same name as the CSV file, and within it, create a directory structure for each slide as follows:

```
DIRECTORY_TO_SAVE_PATCHES/
	CSV_FILE_NAME_{DATE}/
		├── slide_1
    			├── no000000_{coordinate x_0}x_{coordinate y_0}y.jpg
    			├── no000001_{coordinate x_1}x_{coordinate y_1}y.jpg
                ├── ...
                ├── no00000m_{coordinate x_m}x_{coordinate y_m}y.jpg
    			└── thumbnail/
                      └── x20_thumbnail.jpg
		├──slide_2
    			├── no000000_{coordinate x_0}x_{coordinate y_0}y.jpg
    			├── no000001_{coordinate x_1}x_{coordinate y_1}y.jpg
                ├── ...
                ├── no00000m_{coordinate x_m}x_{coordinate y_m}y.jpg
    			└── thumbnail/
                      └── x20_thumbnail.jpg
        ...
		└── slide_N
    			├── no000000_{coordinate x_0}x_{coordinate y_0}y.jpg
    			├── no000001_{coordinate x_1}x_{coordinate y_1}y.jpg
                ├── ...
                ├── no00000m_{coordinate x_m}x_{coordinate y_m}y.jpg
    			└── thumbnail/
                      └── x20_thumbnail.jpg
└── ...
```

**Example from the CPTAC Lung cohort:**
```
DIRECTORY_TO_SAVE_PATCHES/
	CPTAC-LUAD_2025-05-19/
		├── C3L-04365-28
    			├── no000000_000003072x_000006144y.jpg
    			├── no000001_000003072x_000009216y.jpg
    			├── no000002_000006144x_000003072y.jpg
    			├── no000003_000006144x_000006144y.jpg
    			├── no000004_000006144x_000009216y.jpg
    			├── no000005_000009216x_000003072y.jpg
    			├── no000006_000009216x_000006144y.jpg
    			├── no000007_000009216x_000009216y.jpg
    			├── no000008_000012288x_000003072y.jpg
    			├── no000009_000012288x_000006144y.jpg
    			├── no000010_000015360x_000006144y.jpg
    			├── no000011_000018432x_000006144y.jpg
    			└── thumbnail/
                      └── x20_thumbnail.jpg
        └── ...
```

---

***STEP 3 - Extract patch-level features from a patch directory or a csv file containing the list of patch directories.***

**Choice 1 - For extracting patch-level features from a csv file containing the list of patch directories**, you can run the command `scripts/wsi_preprocess/3_run_create_patchdir_list.py` first to generate a csv file containing the list of patch directories as follows:

```bash
cd scripts/wsi_preprocess

python 3_run_create_patchdir_list.py --data_folder ROOT_DIRECTORY_PATH_CONTAINING_PATCHES_FILES --dataset_name DATASET_NAME --save_dir DIRECTORY_TO_SAVE_PATCH_DIR_LIST_CSV_FILE
```

> **--data_folder ROOT_DIRECTORY_PATH_CONTAINING_PATCHES_FILES** is the root directory path where the script will recursively search for folders containing `thumbnail/x20_thumbnail.jpg` files. This thumbnail file is generated as the final step after all patches have been extracted from a WSI. The script identifies valid patch directories by checking for the presence of this thumbnail file, and adds these directories to the list for further processing.

> **--dataset_name DATASET_NAME** is your custom name for the dataset, which will be used in the output CSV filename.

> **--save_dir DIRECTORY_TO_SAVE_PATCH_DIR_LIST_CSV_FILE** is the directory where the CSV file containing the list of patch directories will be saved.

. Then, you can run the command `scripts/wsi_preprocess/4_run_create_wsi_features.py` as follows:

```bash
cd scripts/wsi_preprocess

python 4_run_create_wsi_features.py --batch_size BArTCH_SIZE --model_name FOUNDATION_MODEL_NAME --local_dir False --ckpt model_weight_path --gpu_ids GPU_ID_1 GPU_ID_2 ... --num_processes NUM_PROCESSES --save_dir save_directory --csv_path CSV_FILE_PATH --amp AMP_TYPE --image_loader IMAGE_LOADER_TYPE --image_preprocess PATH_TO_YAML_FILE
```

> **--batch_size BATCH_SIZE** is the batch size for feature extraction.

> **--model_name FOUNDATION_MODEL_NAME** is the name from pathology foundation model list.

> **--local_dir False** is a boolean flag to indicate whether the model checkpoint is located in a local directory.

> **--ckpt model_weight_path** is the path to the model checkpoint file. (required if --local_dir is True)

> **--gpu_ids GPU_ID_1 GPU_ID_2 ...** is the list of GPU IDs to use for feature extraction.

> **--num_processes NUM_PROCESSES** is the number of processes to use for feature extraction.

> **--save_dir save_directory** is the directory where the extracted features will be saved.

> **--csv_path CSV_FILE_PATH** is the path to the CSV file containing the list of patch directories.

> **--amp AMP_TYPE** is the type of AMP (Automatic Mixed Precision) to use for feature extraction. (Choices: fp32, fp16, bf16)

> **--image_loader IMAGE_LOADER_TYPE** is the type of image loader. (Choices: pil, jpeg4py, opencv)

> **--image_preprocess PATH_TO_YAML_FILE** is the path to the YAML file containing image preprocessing configurations. (We provide a default YAML file from `transform_configs/create_patch_feats_transforms.yaml`)

**Example using PLIP model to run the Camelyon+ Dataset in 8 * NVIDIA A100 GPUs with 16 processes:**

```bash
cd scripts/wsi_preprocess

python 4_run_create_wsi_features.py --batch_size 1 --model_name plip --local_dir False --gpu_ids 0 1 2 3 4 5 6 7 --num_processes 16 --save_dir ../WSI_DATA/patch_feature_datasets/CamelyonPlus --csv_path ../WSI_DATA/patchdir_list_csv/CamelyonPlus_2025-05-19.csv --amp fp16 --image_loader pil --image_preprocess ../transform_configs/create_patch_feats_transforms.yaml
```

**Choice 2 - For extracting patch-level features from a patch directory**, you can run the command `scripts/wsi_preprocess/4_run_create_patch_features.py` directly as follows:

```bash
cd scripts/wsi_preprocess

python 4_run_create_patch_features.py --batch_size BATCH_SIZE --model_name FOUNDATION_MODEL_NAME --local_dir False --ckpt model_weight_path --gpu_ids GPU_ID_1 GPU_ID_2 ... --num_processes NUM_PROCESSES --save_dir save_directory --patch_slide_dir PATH_TO_PATCH_DIRECTORY --amp AMP_TYPE --image_loader IMAGE_LOADER_TYPE --image_preprocess PATH_TO_YAML_FILE
```

> **--patch_slide_dir PATH_TO_PATCH_DIRECTORY** is the path to the patch directory. We provide an example for PATH_TO_PATCH_DIRECTORY as follows:

> ```
>PATH_TO_PATCH_DIRECTORY/
>	slide_1/
>		no000000_000003072x_000006144y.jpg
>		no000001_000003072x_000009216y.jpg
>		...
>		thumbnail/
>			x20_thumbnail.jpg
>	slide_2/
>		no000000_000003072x_000006144y.jpg
>		no000001_000003072x_000009216y.jpg
>		...
>		thumbnail/
>			x20_thumbnail.jpg
>	...
> ```



### 🌞 *3. Fine-tuning patch-level tasks*
Will be released on the last week on May 2025! (Linear Probing, Full Parameter, PathFiT, etc.)

### ⭐ *4. Fine-tuning slide-level (WSI) tasks.*

First we need to generate `json` file for the slide-level downstream tasks. The json file should be in the following format:

```json
{
	"train": [
		{
			"feat_path": "slide_1.pth",
			"label": "label_1",
			"patch_dir": "PATH_TO_PATCH_DIRECTORY/slide_1"
		},
		{
			"feat_path": "slide_2.pth",
			"label": "label_2",
			"patch_dir": "PATH_TO_PATCH_DIRECTORY/slide_2"
		},
		{
			"feat_path": "slide_3.pth",
			"label": "label_3",
			"patch_dir": "PATH_TO_PATCH_DIRECTORY/slide_3"
		}
		...
	],
	"valid": [
		{
			"feat_path": "slide_4.pth",
			"label": "label_4",
			"patch_dir": "PATH_TO_PATCH_DIRECTORY/slide_4"
		}
		...
	],
	"test": [
		{
			"feat_path": "slide_5.pth",
			"label": "label_5",
			"patch_dir": "PATH_TO_PATCH_DIRECTORY/slide_5"
		}
		...
	]
}
```
Then, we can run the command `scripts/wsi_classification/run_wsi_train.py` to fine-tune the slide-level downstream tasks.

```bash
cd scripts/wsi_classification

python run_wsi_train.py --data_json PATH_TO_JSON_FILE --model_name FOUNDATION_MODEL_NAME --training_mode MIL_BASELINE_METHOD --batch_size 1 --num_epochs 15 --lr 1e-4 --weight_decay 1e-4 --seed 42 --amp_dtype bfloat16 --save_metric bal_accuracy --save_interval 5 --save_dir DIRECTORY_TO_SAVE_MODEL --gpu_ids GPU_ID --few_shot FEW_SHOT_NUMBER
```

> **--data_json PATH_TO_JSON_FILE** is the path to the json file containing the list of `feat_path`, `label`, and `patch_dir` (optional).

> **--model_name FOUNDATION_MODEL_NAME** is the name from pathology foundation model list.

> **--training_mode MIL_BASELINE_METHOD** is the multiple instance learning (MIL) baseline method to use for the slide-level downstream tasks. (Choices: abmil, simlp)

> **--batch_size BATCH_SIZE** is the batch size for training. For all the baseline methods, we recommend using batch size 1.

> **--num_epochs NUM_EPOCHS** is the number of epochs to train. Default: 15.

> **--lr LEARNING_RATE** is the learning rate for training. Default: 1e-4.

> **--weight_decay WEIGHT_DECAY** is the weight decay for training. Default: 1e-4.

> **--seed SEED** is the seed for training. Default: 42.

> **--amp_dtype AMP_TYPE** is the type of AMP (Automatic Mixed Precision) to use for training. (Choices: float32, float16, bfloat16). Default: bfloat16.

> **--save_metric BAL_ACCURACY** is the metric to save the model. (Choices: bal_accuracy, accuracy, auc, f1, kappa). Default: bal_accuracy.

> **--save_interval SAVE_INTERVAL** is the interval to save the model. Default: 5.

> **--save_dir DIRECTORY_TO_SAVE_MODEL** is the directory to save the model. 

> **--gpu_ids GPU_ID** is the GPU ID to use for training.

> **--few_shot FEW_SHOT_NUMBER** is the number of few-shot samples to use for training. Default: None. If provided, the model will be trained with the few-shot samples.


After training, we can run the command `scripts/wsi_classification/run_wsi_infer.py` to test the model.

```bash
cd scripts/wsi_classification

python run_wsi_infer.py --test_json PATH_TO_JSON_FILE --model_name FOUNDATION_MODEL_NAME --training_mode MIL_BASELINE_METHOD --finetune_ckpt PATH_TO_SAVE_MODEL_CHECKPOINT --gpu_ids GPU_ID --output_dir DIRECTORY_TO_SAVE_INFERENCE_RESULT --plot_confusion True
```

> **--test_json PATH_TO_JSON_FILE** is the path to the json file containing the list of `feat_path`, `label`, and `patch_dir` (optional).

> **--model_name FOUNDATION_MODEL_NAME** is the name from pathology foundation model list.

> **--training_mode MIL_BASELINE_METHOD** is the multiple instance learning (MIL) baseline method to use for the slide-level downstream tasks. (Choices: abmil, simlp)

> **--finetune_ckpt PATH_TO_SAVE_MODEL_CHECKPOINT** is the path to the saved model checkpoint file. Our pipeline will only save the checkpoint of model classifier head, which is lightweight and can be loaded quickly.

> **--gpu_ids GPU_ID** is the GPU ID to use for inference.

> **--output_dir DIRECTORY_TO_SAVE_INFERENCE_RESULT** is the directory to save the inference result. If not provided, the default directory will be the same as the finetuned model checkpoint directory.

> **--plot_confusion True** is a boolean flag to indicate whether to plot the confusion matrix.

---

### Acknowledgements

We would like to express our sincere gratitude to the creators of all the foundation models and MIL methods used in this project. If you have any questions, feel free to contact us at the provided email 
(lingxt23@mails.tsinghua.edu.cn) and (jw-li24@mails.tsinghua.edu.cn).

--- 
***H&G Pathology AI Research Team, Tsinghua University***




