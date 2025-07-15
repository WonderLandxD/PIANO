MODEL_HF_PATHS = {
    "openai_clip_p16": "openai/clip-vit-base-patch16",  
    "plip": "vinid/plip",  
    "conch_v1": "hf_hub:MahmoodLab/conch",
    "conch_v1_5": "MahmoodLab/TITAN",
    "uni_v1": "hf-hub:MahmoodLab/uni",
    "uni_v2": "hf-hub:MahmoodLab/UNI2-h",
    "prov_gigapath": "hf_hub:prov-gigapath/prov-gigapath",
    "virchow_v1": "hf-hub:paige-ai/Virchow",
    "virchow_v2": "hf-hub:paige-ai/Virchow2",
    "musk": "hf_hub:xiangjx/musk",
    "h_optimus_0": "hf-hub:bioptimus/H-optimus-0",
    "h_optimus_1": "hf-hub:bioptimus/H-optimus-1",
    "phikon_v1": "owkin/phikon",
    "phikon_v2": "owkin/phikon-v2",
    "quiltnet_b_32": "hf-hub:wisdomik/QuiltNet-B-32",
    "quiltnet_b_16": "hf-hub:wisdomik/QuiltNet-B-16",
    "quiltnet_b_16_pmb": "hf-hub:wisdomik/QuiltNet-B-16-PMB",
    "ctranspath": "JWonderLand/CHIEF_unofficial",
    "dino_hipt": "JWonderLand/HIPT_unofficial",
    "beph": "JWonderLand/BEPH_unofficial",
    "pathorchestra": "hf-hub:AI4Pathology/PathOrchestra"
}


def get_model_output_dim(model_name: str) -> int:
    """
    Get the output dimension of a model by its name.

    Args:
        model_name (str): Name of the model

    Returns:
        int: Output dimension of the model

    """
    output_dims = {
        "openai_clip_p16": 512,
        "plip": 512,
        "conch_v1": 512,
        "conch_v1_5": 768,
        "uni_v1": 1024,
        "uni_v2": 1536,
        "prov_gigapath": 1536,
        "virchow_v1": 2560,
        "virchow_v2": 2560,
        "musk": 2048,
        "h_optimus_0": 1536,
        "h_optimus_1": 1536,
        "phikon_v1": 768,
        "phikon_v2": 768,
        "quiltnet_b_32": 512,
        "quiltnet_b_16": 512,
        "quiltnet_b_16_pmb": 512,
        "ctranspath": 768,
        "dino_hipt": 384,
        "beph": 768,
        "pathorchestra": 1024
    }

    if model_name not in output_dims:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(output_dims.keys())}")
    
    return output_dims[model_name]


def get_model_hf_path(model_name):
    """
    return the huggingface path for the model
    """
    if model_name not in MODEL_HF_PATHS:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_HF_PATHS.keys())}")
    return MODEL_HF_PATHS[model_name] 