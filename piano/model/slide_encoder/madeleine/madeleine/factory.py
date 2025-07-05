import os
import json
from argparse import Namespace

from .Model import create_model
import pdb

from huggingface_hub import snapshot_download, hf_hub_download


# local_dir = ''
# snap_download(repo_id="hf_hub:MahmoodLab/madeleine", local_dir=local_dir)
# create_model_from_pretrained(local_dir)

def create_model_from_pretrained(local_dir: str = None):
    

    if local_dir is None:
        model_cfg = hf_hub_download(repo_id="MahmoodLab/madeleine", filename="model_config.json")
        checkpoint_path = hf_hub_download(repo_id="MahmoodLab/madeleine", filename="model.pt")
    else:
        model_cfg = os.path.join(local_dir, "model_config.json")
        checkpoint_path = os.path.join(local_dir, "model.pt")

    # load config and weights 
    model_cfg = Namespace(**model_cfg)  
    
    model = create_model(
        model_cfg,
        device="cuda",
        checkpoint_path=checkpoint_path,
    )

    return model


# test create_model_from_pretrained
if __name__ == "__main__":
    
    path_to_model = "../../results_brca/dfc80197ddc463b89ee1cd2a5d89f421"
    ckpt_path = os.path.join(path_to_model, "model.pt")
    config_path = os.path.join(path_to_model, "config.json")
    
    # load config and change to namespace 
    
    with open(config_path) as f:
        config = json.load(f)
    
    # convert json to name space
    config = Namespace(**config)
    
    # load model 
    model = create_model_from_pretrained(
        model_cfg=config,
        device='cuda',
        checkpoint_path=ckpt_path,
    )
    
    pdb.set_trace()