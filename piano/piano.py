"""
PIANO - Pathology AI Network for Outcome prediction
Main interface module that provides unified access to patch encoders and MIL models.
"""

# Import patch encoder functions
from .model.patch_encoder import create_patch_encoder, get_model_output_dim, get_model_hf_path

# Import MIL model functions  
from .model.mil_factory import create_mil_model


# Export all main functions
__all__ = [
    'create_model',
    'create_patch_encoder', 
    'create_mil_model',
    'get_model_output_dim',
    'get_model_hf_path'
] 