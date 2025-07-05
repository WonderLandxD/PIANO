import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

# Import base model and registry
from .base_model import BaseModel, MODEL_REGISTRY
from .model_registry import get_model_output_dim, get_model_hf_path

# Import all model modules to register them
from .plip.plip_model import PLIPModel, OpenAICLIPModel
from .uni.uni_v1_model import UNIModel
from .uni.uni_v2_model import UNI2Model
from .conch.conch_v1_model import CONCHModel
from .conch.conch_v1_5_model import CONCHV1_5Model
from .virchow.virchow_v1_model import VirchowModel
from .virchow.virchow_v2_model import Virchow2Model
from .prov_gigapath.prov_gigapath_model import ProvGigaPathModel
from .ctranspath.ctranspath_model import CTransPathModel
from .quiltnet.quiltnet_models import QuiltNetB32Model, QuiltNetB16Model, QuiltNetB16_PMB_Model
from .phikon.phikon_models import PhikonV1Model, PhikonV2Model
from .h_optimus.h_optimus_models import HOptimus0Model, HOptimus1Model
from .musk.musk_model import MuskModel
from .dino_hipt.dino_hipt_model import HIPTDinoModel
from .beph.beph_model import BEPHModel
from .pathorchestra.pathorchestra_model import PathOrchestraModel


def create_patch_encoder(model_name, checkpoint_path=None, local_dir=False):
    """
    Create a pathology foundation model.

    Args:
        model_name (str): Name of the model to create
        checkpoint_path (str, optional): Path to model checkpoint. Defaults to None.
        local_dir (bool, optional): Whether checkpoint_path is a local directory. Defaults to False.

    Returns:
        BaseModel: Initialized model instance

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(checkpoint_path=checkpoint_path, local_dir=local_dir)

    return model


# Export key functions and classes
__all__ = [
    'create_patch_encoder',
    'get_model_output_dim', 
    'get_model_hf_path',
    'BaseModel',
    'MODEL_REGISTRY'
] 