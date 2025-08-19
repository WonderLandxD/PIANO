def COBRA2_Model(*args, **kwargs):
    """Lazy loading function for COBRA2_Model to avoid import errors"""
    try:
        from .cobra_model import COBRA2_Model as _COBRA2_Model
        return _COBRA2_Model(*args, **kwargs)
    except ImportError as e:
        raise ImportError(f"COBRA2 model requires 'mamba_ssm' library which is not installed. Please install it using: pip install mamba-ssm. Error: {e}")

__all__ = ['COBRA2_Model']