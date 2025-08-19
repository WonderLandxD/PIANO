def create_slide_encoder(model_name, local_dir: str = None):
    """
    Create a slide encoder model with lazy loading to avoid dependency issues.
    
    Args:
        model_name (str): Name of the model ('chief', 'gigapath', 'titan', 'prism', 'madeleine', 'cobra', etc.)
        local_dir (str): Local directory path for model weights (optional)
    
    Returns:
        torch.nn.Module: The slide encoder model
    """
    if model_name == 'chief':
        from .chief import CHIEFModel
        return CHIEFModel(local_dir=local_dir)
    elif model_name == 'gigapath':
        from .gigapath import GigaPathModel
        return GigaPathModel(local_dir=local_dir)
    elif model_name == 'titan':
        from .titan import TITANModel
        return TITANModel()
    elif model_name == 'prism':
        from .prism import PRISMModel
        return PRISMModel()
    elif model_name == 'madeleine':
        from .madeleine import MADELEINEModel
        return MADELEINEModel(local_dir=local_dir)
    elif model_name == 'hipt':
        from .hipt import HIPTModel
        return HIPTModel()
    elif model_name == 'feather_uni_v1':
        from .feather import FEATHER_UNIV1_Model
        return FEATHER_UNIV1_Model()
    elif model_name == 'protea':
        from .protea import PROTEAModel
        return PROTEAModel()
    elif model_name == 'cobra':
        try:
            from .cobra import COBRA2_Model
            return COBRA2_Model()
        except ImportError as e:
            raise ImportError(f"COBRA model requires additional dependencies that are not installed: {e}")
    else:
        raise ValueError(f"Invalid model name: {model_name}")


__all__ = ['create_slide_encoder'] 