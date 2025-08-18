from .chief import CHIEFModel
from .gigapath import GigaPathModel
from .titan import TITANModel
from .prism import PRISMModel
from .madeleine import MADELEINEModel
from .feather import FEATHER_UNIV1_Model
from .protea import PROTEAModel
from .hipt import HIPTModel
from .cobra import COBRA2_Model


def create_slide_encoder(model_name, local_dir: str = None):
    """
    Create a slide encoder model.
    
    Args:
        model_name (str): Name of the model ('chief', 'gigapath', 'titan', 'prism', 'madeleine')
        local_dir (str): Local directory path for model weights (optional)
    
    Returns:
        torch.nn.Module: The slide encoder model
    """
    if model_name == 'chief':
        return CHIEFModel(local_dir=local_dir)
    elif model_name == 'gigapath':
        return GigaPathModel(local_dir=local_dir)
    elif model_name == 'titan':
        return TITANModel()
    elif model_name == 'prism':
        return PRISMModel()
    elif model_name == 'madeleine':
        return MADELEINEModel(local_dir=local_dir)
    elif model_name == 'hipt':
        return HIPTModel()
    elif model_name == 'feather_uni_v1':
        return FEATHER_UNIV1_Model()
    elif model_name == 'protea':
        return PROTEAModel()
    elif model_name == 'cobra':
        return COBRA2_Model()
    else:
        raise ValueError(f"Invalid model name: {model_name}")


__all__ = ['create_slide_encoder', 'CHIEFModel', 'GigaPathModel', 'TITANModel', 'PRISMModel', 'MADELEINEModel', 'FEATHER_UNIV1_Model'] 