# 🎶 PIANO: Pathology Image ANalysis Orchestrator 

**PIANO** is a simple PyTorch library for pathology image analysis. It helps you generate patches from whole-slide images, use pathology foundation models for feature extraction, and more! 🚀

**Features ✨**

•	🖼️ **Whole-slide image processing**: Load and preprocess whole slide images easily.

•	🧩 **Patch generation**: Automatically generate patches for model training.

•	🧠 **Foundation models**: Use pre-trained pathology foundation models to extract powerful features.

•	🌍 More functionality will be released in the future.

---------

## 📰 News

**2025-03-18:** Enhanced model flexibility with local loading support, added `CTransPath (CHIEF-based weights)` model, and introduced YAML-based preprocessing customization and coordinate saving.

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

### 🪐 *1. Generating patches from histopathology whole slide images (WSIs).*
To batch-generate patches for WSIs, we first run the command `run_create_wsi_list.py` as follows:

```
python run_create_wsi_list.py --data_folder Root_directory_path_containing_WSI_files --dataset_name Dataset_name --save_dir Directory_to_save_CSV_file
```

Then run the command `run_generate_patches.py` as follows. We recommend using **8** processes since it works on regular CPU:

```
python run_generate_patches.py --n_thread Number_of_threads_to_use --csv_path Path_to_the_CSV_file --save_dir Directory_to_save_patches
```

### 🌕 *2. Define a pretrained pathology foundation model.*

We use [PLIP](https://www.nature.com/articles/s41591-023-02504-3) as an example.

```bash
# Set up Hugging Face mirror (if you encounter issues downloading models)
export HF_ENDPOINT='https://hf-mirror.com'

# Log in to Hugging Face (replace with your write token)
huggingface-cli login --token <huggingface_write_token>

# Download the model and customize the path
# Set the model name and custom path
export MODEL_NAME='vinid/plip'
export CUSTOM_PATH='/your/custom/path'

# Use huggingface-cli to download the model to the custom path
huggingface-cli download $MODEL_NAME --cache-dir $CUSTOM_PATH
```

```python
# Load the PLIP model from Hugging Face
import piano
model = piano.create_model("plip").cuda()
```

```python
# Load the PLIP model from local path
model = piano.create_model("plip", checkpoint_path="/your/custom/path", local_dir=True).cuda()
```

```python
# Load the image preprocess and text preprocess from the original model (not all models have text preprocess)
image_preprocess = model.image_preprocess
text_preprocess = model.text_preprocess
```

•	**model:** The pathology foundation model.

•	**image_preprocess:** Preprocessing function for image input (scaling, normalization, etc.).

•	**text_preprocess:** Preprocessing function for text input (tokenization, padding, etc.).

### Example Usage of all codes:

```python
# Import necessary libraries
import torch
from PIL import Image
import numpy as np
import piano

# 1. Define the model
# Load the PLIP model from Hugging Face
model = piano.create_model("plip").cuda()

# Or load the model from a local path
# model = piano.create_model("plip", checkpoint_path="/your/custom/path", local_dir=True).cuda()

# Get the model's preprocessing functions
image_preprocess = model.image_preprocess
text_preprocess = model.text_preprocess

# 2. Load the image
image = Image.open('./img/sample_lusc.jpg')  # 256px * 256px resolution
text_labels = ["lung adenocarcinoma", "lung squamous cell carcinoma", "normal"]  # Candidate text labels

# 3. Use the model's preprocessing functions to process the image and text
image_tensor = image_preprocess(image).unsqueeze(0).cuda()  # [1, 3, 256, 256]
text_tensors = [text_preprocess([label]).unsqueeze(0).cuda() for label in text_labels]  # Each label [1, 77]

# 4. Extract features
model.eval()
with torch.inference_mode():
    # Encode image features
    img_feat = model.encode_image(image_tensor)  # [1, C]
    
    # Encode all text features
    text_feats = torch.cat([model.encode_text(text_tensor) for text_tensor in text_tensors])  # [N, C], N is the number of labels

    # 6. Calculate the similarity between the image features and each text feature
    similarities = torch.nn.functional.cosine_similarity(img_feat, text_feats, dim=1)  # [N]

    # 7. Find the most similar label
    best_match_idx = torch.argmax(similarities).item()
    best_match_label = text_labels[best_match_idx]

# 8. Output the result
print(f"The image most likely belongs to the category: {best_match_label}")
print(f"Similarity scores for all categories: {dict(zip(text_labels, similarities.cpu().numpy()))}")
```

•	**model.encode_image:** Extracts features from the image.

•	**model.encode_text:** Extracts features from the text.

  - For **visual-only models**, `model.encode_image` works *without normalization*, but `text_preprocess` and `model.encode_text` will be *inactive*.
  - For **visual-language models**, both `model.encode_image` and `model.encode_text` are processed *with normalization (F.normalize)*.

**Additional Notes:**

- `model.backbone`: This is the raw model used for feature extraction. You can access it directly if needed.
- `model.get_img_token`: Returns the output, which contains output['patch_tokens'] and output['class_token'].
- `piano.get_model_hf_path(model_name)`: Returns the Hugging Face model path for a given model name. This is useful if you want to load a model checkpoint from Hugging Face.

### Model Checkpoint Paths and Sources
| **Model Name**           | **Model Checkpoint Path**          | **Link for `piano.get_model_hf_path(model_name)`**                             |
|--------------------------|------------------------------------|-----------------------------------------------------------------------------------|
| `plip`                   | `vinid/plip`                       | [Hugging Face - vinid/plip](https://huggingface.co/vinid/plip)                    |
| `openai_clip_p16`        | `openai/clip-vit-base-patch16`     | [Hugging Face - openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16) |
| `conch_v1`               | `hf_hub:MahmoodLab/conch`          | [Hugging Face - MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH) |
| `uni_v1`                 | `hf-hub:MahmoodLab/uni`            | [Hugging Face - MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) |
| `uni_v2`                 | `hf-hub:MahmoodLab/UNI2-h`         | [Hugging Face - MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) |
| `prov_gigapath`          | `hf_hub:prov-gigapath/prov-gigapath` | [Hugging Face - prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) |
| `virchow_v1`             | `hf-hub:paige-ai/Virchow`          | [Hugging Face - paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow) |
| `virchow_v2`             | `hf-hub:paige-ai/Virchow2`         | [Hugging Face - paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) |
| `musk`                   | `hf_hub:xiangjx/musk`              | [Hugging Face - xiangjx/musk](https://huggingface.co/xiangjx/musk) |
| `h_optimus_0`            | `hf-hub:bioptimus/H-optimus-0`     | [Hugging Face - bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) |
| `ctranspath`             | `YOUR_LOCAL_PATH`                  | --- |

**Note**: `local_dir` means the path to your local model files downloaded from the official source.

*More models will be supported (Updated on 03-18-2025).*

### 🌞 *3. Using pathology foundation models to create patch features.*

```
python run_create_features.py \
    --batch_size 1 \
    --model_name choose_one_foundation_model \
    --ckpt model_weight_path \
    --gpu_num 1 \
    --save_dir save_directory \
    --patch_slide_dir patches_derectory \
    --image_preprocess image_preprocess.yaml
```

- **batch_size:** (int) Batch size for feature extraction. Default is 1.
- **model_name:** (str) The name of the pathology foundation model to use for feature extraction (e.g., "plip", "openai_clip_p16"). This is a required argument.
- **ckpt:** (str) Path to the checkpoint file for the chosen model. This is a required argument.
- **gpu_num:** (int) The number of GPUs to use. Default is 1.
- **num_workers:** (int) Number of workers for loading data. Default is 1.
- **save_dir:** (str) Directory where the extracted features will be saved. This is a required argument.
- **patch_slide_dir:** (str) Directory containing the patches (cropped from slides). This is a required argument.
- **image_preprocess:** (str) Path to the YAML file containing image preprocessing configurations.




