# 🎶 PIANO: Pathology Image Analysis Orchestrator 

**PIANO** is a simple PyTorch library for pathology image analysis. It helps you generate patches from whole-slide images, use pathology foundation models for feature extraction, and more! 🚀

**Features ✨**

•	🖼️ **Whole-slide image processing**: Load and preprocess whole slide images easily.

•	🧩 **Patch generation**: Automatically generate patches for model training.

•	🧠 **Foundation models**: Use pre-trained pathology foundation models to extract powerful features.

•	🌍 More functionality will be released in the future.

---------

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

•	model_name='plip': Specifies the foundation model.

•	checkpoint_path='vinid/plip': Path to the pretrained PLIP model checkpoint (from HuggingFace or local path).

•	device='cuda:0': Choose the device to run the model (e.g., cuda, cuda:0, cuda:1, or CPU).

```
# Set the Hugging Face endpoint to the mirror if you have trobule downloading the model
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Login to Hugging Face
from huggingface_hub import login
login(<huggingface write token>)
```

```
# Load the PLIP model and preprocessing functions

model, image_preprocess, text_preprocess = piano.create_piano(
    model_name='plip', 
    checkpoint_path='vinid/plip', 
    device='cuda:0'
)
```

•	model: The foundation model used for feature extraction.

•	image_preprocess: Preprocessing function for image input (scaling, normalization, etc.).

•	text_preprocess: Preprocessing function for text input (tokenization, padding, etc.).

### Example Usage:

```
from PIL import Image

# Load an image
image = Image.open('path_to_image.jpg') # 256px * 256px resolution
text = ["LUSC", "LUAD"]

# Preprocess the image and text
image_tensor = image_preprocess(image).unsqueeze(0).cuda()  # [1, 3, 256, 256]
text_tensor = text_preprocess(text).unsqueeze(0).cuda()  # [2, 77]

model.set_mode('eval')
with torch.inference_mode():
    # Encode image and text features
    img_feat = model.encode_image(image_tensor) # [1, C]
    text_feat = model.encode_text(text_tensor) # [2, C]
```

•	model.encode_image: Extracts features from the image.

•	model.encode_text: Extracts features from the text.

  - For **visual-only models**, `model.encode_image` works *without normalization*, but `text_preprocess` and `model.encode_text` will be *inactive*.
  - For **visual-language models**, both `model.encode_image` and `model.encode_text` are processed *with normalization (F.normalize)*.

**Additional Notes:**

- `model.backbone`: This is the raw model used for feature extraction. You can access it directly if needed.
- `image_preprocess`: These are the valid preprocessing (transform) functions that prepare the batch image before passing it to the model.
- `text_preprocess`:These are the preprocessing functions that prepare the text list before passing it to the model.
- `model.forward`: This is the function that handles the actual image feature extraction, performing forward propagation in the network.
- `model.set_mode('eval')`: Sets the model to evaluation mode. This is necessary when you’re testing or inferring with the model, ensuring dropout layers (if any) are turned off.
- `model.set_mode('train')`: Sets the model to training mode. This enables dropout layers and other training-specific behaviour.
- `piano.get_model_hf_path(model_name)`: Returns the Hugging Face model path for a given model name. This is useful if you want to load a model checkpoint from Hugging Face.

### Model Checkpoint Paths and Sources
| **Model Name**           | **Model Checkpoint Path**          | **Link for `piano.get_model_hf_path(model_name)`**                             |
|--------------------------|------------------------------------|-----------------------------------------------------------------------------------|
| `plip`                   | `vinid/plip`                       | [Hugging Face - vinid/plip](https://huggingface.co/vinid/plip)                    |
| `openai_clip_p16`        | `openai/clip-vit-base-patch16`     | [Hugging Face - openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16) |
| `conch_v1`               | `local_dir`                        | [Hugging Face - MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH) |
| `uni_v1`                  | `hf-hub:MahmoodLab/uni`                        | [Hugging Face - MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) |
| `uni_v2`                  | `hf-hub:MahmoodLab/UNI2-h`                        | [Hugging Face - MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) |
| `prov_gigapath_tile`      | `hf_hub:prov-gigapath/prov-gigapath` | [Hugging Face - prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) |
| `virchow_v1`              | `hf_hub:paige-ai/Virchow`            | [Hugging Face - paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow) |
| `virchow_v2`              | `hf_hub:paige-ai/Virchow2`           | [Hugging Face - paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) |
| `musk`                     | `hf_hub:xiangjx/musk`                | [Hugging Face - xiangjx/musk](https://huggingface.co/xiangjx/musk) |
| `h_optimus_0`              | `hf_hub:bioptimus/H-optimus-0`        | [Hugging Face - bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) |

**Note**: `local_dir` means the path to your local model files downloaded from the official source. 

*More models will be supported (Updated on 01-31-2025).*

### 🌞 *3. Using pathology foundation models to create patch features.*

```
python run_create_features.py \
    --batch_size 1 \
    --model_name choose_one_foundation_model \
    --ckpt model_weight_path \
    --gpu_num 1 \
    --save_dir save_directory \
    --patch_slide_dir patches_derectory
```

- batch_size: (int) Batch size for feature extraction. Default is 1.
- model_name: (str) The name of the pathology foundation model to use for feature extraction (e.g., "plip", "openai_clip_p16"). This is a required argument.
- ckpt: (str) Path to the checkpoint file for the chosen model. This is a required argument.
- gpu_num: (int) The number of GPUs to use. Default is 1.
- num_workers: (int) Number of workers for loading data. Default is 1.
- save_dir: (str) Directory where the extracted features will be saved. This is a required argument.
- patch_slide_dir: (str) Directory containing the patches (cropped from slides). This is a required argument.




