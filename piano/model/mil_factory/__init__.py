"""
MIL Factory - Multiple Instance Learning Models Factory

This module provides a unified interface for creating MIL baseline models.
Uses lazy loading to avoid import failures affecting other models.
"""

import torch
import torch.nn as nn

# ============================================================================
# Part 1: Available Models Registry (Lazy Loading)
# ============================================================================

# Available model names for lazy loading
AVAILABLE_MODELS = [
    'abmil', 'gated_abmil', 'clam_sb', 'clam_mb', 'transmil', 'dgrmil',
    'dtfdmil', 'dsmil', 'ilramil', 'wikg', 's4mil', 'amdmil', 
    'mean_pool', 'max_pool', '2dmamba', 'mambamil', 'm4'
]

def _lazy_load_model(model_name):
    """
    Lazy load model class based on model name
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        class: Model class
        
    Raises:
        ImportError: If model cannot be imported
        ValueError: If model name is not recognized
    """
    if model_name == 'abmil':
        from .abmil.abmil import ABMIL
        return ABMIL
    elif model_name == 'gated_abmil':
        from .abmil.abmil import GatedABMIL
        return GatedABMIL
    elif model_name == 'clam_sb':
        from .clam.clam import CLAM_SB
        return CLAM_SB
    elif model_name == 'clam_mb':
        from .clam.clam import CLAM_MB
        return CLAM_MB
    elif model_name == 'transmil':
        from .transmil.transmil import TransMIL
        return TransMIL
    elif model_name == 'dgrmil':
        from .dgrmil.dgrmil import DGRMIL
        return DGRMIL
    elif model_name == 'dtfdmil':
        from .dtfdmil.dtfdmil import DTFDMIL
        return DTFDMIL
    elif model_name == 'dsmil':
        from .dsmil.dsmil import DSMIL
        return DSMIL
    elif model_name == 'ilramil':
        from .ilramil.ilramil import ILRAMIL
        return ILRAMIL
    elif model_name == 'wikg':
        from .wikg.wikg import WiKG
        return WiKG
    elif model_name == 's4mil':
        from .s4mil.s4mil import S4MIL
        return S4MIL
    elif model_name == 'amdmil':
        from .amdmil.amdmil import AMD_MIL
        return AMD_MIL
    elif model_name == 'mean_pool':
        from .pooling.mil import MeanPool
        return MeanPool
    elif model_name == 'max_pool':
        from .pooling.mil import MaxPool
        return MaxPool
    elif model_name == '2dmamba':
        from .mamba_2d.mamba_2d import MambaMIL_2D
        return MambaMIL_2D
    elif model_name == 'mambamil':
        from .mambamil.mambamil import MambaMIL
        return MambaMIL
    elif model_name == 'm4':
        from .m4.m4 import M4
        return M4
    else:
        raise ValueError(f"Unknown model: {model_name}. Available models: {AVAILABLE_MODELS}")

# ============================================================================
# Part 2: Model Registry (For compatibility)
# ============================================================================

class LazyModelRegistry:
    """Lazy model registry that loads models on demand"""
    
    def __contains__(self, key):
        return key in AVAILABLE_MODELS
    
    def __getitem__(self, key):
        return _lazy_load_model(key)
    
    def keys(self):
        return AVAILABLE_MODELS

# Create registry instance
MIL_MODEL_REGISTRY = LazyModelRegistry()

# ============================================================================
# Part 3: Default Parameters for each model
# ============================================================================

MIL_DEFAULT_PARAMS = {
    'abmil': {'dim_in': 1024, 'dim_hidden': 512, 'num_classes': 2, 'dropout': 0.25, 'survival': False},
    'gated_abmil': {'dim_in': 1024, 'dim_hidden': 512, 'num_classes': 2, 'dropout': 0.25, 'survival': False},
    'clam_sb': {'dim_in': 1024, 'dim_hidden': 512, 'dropout': 0.25, 'num_classes': 2, 'k_sample': 8, 'instance_loss_fn': None, 'subtyping': False, 'survival': False},
    'clam_mb': {'dim_in': 1024, 'dim_hidden': 512, 'dropout': 0.25, 'num_classes': 2, 'k_sample': 8, 'instance_loss_fn': None, 'subtyping': False, 'survival': False},
    'dgrmil': {'dim_in': 1024, 'num_classes': 2, 'L': 512, 'D': 128, 'n_lesion': 11, 'attn_mode': 'gated', 'dropout_node': 0.0, 'dropout_patch': 0.0, 'initialize': False, 'survival': False},
    'dsmil': {'dim_in': 1024, 'num_classes': 2, 'nonlinear': True, 'survival': False},
    'dtfdmil': {'dim_in': 1024, 'dim_hidden': 512, 'num_classes': 2, 'num_groups': 4, 'numLayer_Res': 0, 'classifier_dropout': 0.25, 'attCls_dropout': 0.25, 'distill': 'MaxMinS', 'survival': False},
    'ilramil': {'dim_in': 1024, 'dim_hidden': 512, 'num_classes': 2, 'dropout': 0.25, 'num_layers': 2, 'num_heads': 8, 'topk': 2, 'ln': False, 'survival': False},
    's4mil': {'dim_in': 1024, 'num_classes': 2, 'dropout': 0.25, 'act': 'gelu', 'survival': False},
    'transmil': {'dim_in': 1024, 'dim_hidden': 512, 'num_classes': 2, 'num_layers': 2, 'num_heads': 8, 'dropout': 0.25, 'survival': False},
    'wikg': {'dim_in': 384, 'dim_hidden': None, 'num_classes': 2, 'topk': 6, 'agg_type': 'bi-interaction', 'dropout': 0.3, 'pool': 'attn', 'survival': False},
    'mean_pool': {'dim_in': 1024, 'num_classes': 2, 'survival': False},
    'max_pool': {'dim_in': 1024, 'num_classes': 2, 'survival': False},
    'amdmil': {'dim_in': 1024, 'embed_dim': 512, 'num_classes': 10, 'agent_num': 256, 'survival': False}, 
    '2dmamba': {'dim_in': 1024, 'drop_out': 0.25, 'num_classes': 2, 'survival': False, 'pos_emb_type': None},
    'mambamil': {'dim_in': 1024, 'num_classes': 2, 'dropout': 0.25, 'act': 'gelu', 'survival': False, 'layer': 2, 'rate': 10, 'type': 'SRMamba'},
    'm4': {'dim_in': 1024, 'experts_out': 512, 'towers_out': 2, 'num_classes': 2, 'towers_hidden': 128, 'tasks': 1, 'num_expert': 4, 'survival': False},
}

# ============================================================================
# Part 4: Model Creation Function
# ============================================================================

def create_mil_model(model_name, **kwargs):
    """
    Create MIL model using lazy loading
    
    Args:
        model_name (str): Model name
        **kwargs: Model parameters that will override default parameters
        
    Returns:
        torch.nn.Module: Created model instance
        
    Raises:
        ValueError: If model_name is not recognized
        ImportError: If model cannot be imported
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {AVAILABLE_MODELS}")
    
    try:
        # Lazy load the model class
        model_class = _lazy_load_model(model_name)
    except ImportError as e:
        raise ImportError(f"Failed to import model '{model_name}': {e}")
    
    # Get default parameters for the model
    if model_name in MIL_DEFAULT_PARAMS:
        default_kwargs = MIL_DEFAULT_PARAMS[model_name].copy()
    else:
        # Fallback for any models not in default params
        default_kwargs = {'dim_in': 1024, 'num_classes': 2, 'survival': False}
    
    # Update default parameters with passed kwargs
    default_kwargs.update(kwargs)
    
    return model_class(**default_kwargs)


def get_mil_model_names():
    """
    Get all available MIL model names
    
    Returns:
        list: List of available model names
    """
    return AVAILABLE_MODELS.copy()


def get_mil_default_params(mil_name):
    """
    Get default parameters for a specific MIL model
    
    Args:
        mil_name (str): Model name
        
    Returns:
        dict: Default parameters for the model
        
    Raises:
        ValueError: If mil_name is not recognized
    """
    if mil_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {mil_name}. Available models: {AVAILABLE_MODELS}")
    
    if mil_name in MIL_DEFAULT_PARAMS:
        return MIL_DEFAULT_PARAMS[mil_name].copy()
    else:
        return {'dim_in': 1024, 'num_classes': 2, 'survival': False}


# ============================================================================
# Part 5: Export
# ============================================================================

__all__ = [
    'create_mil_model',
    'get_mil_model_names', 
    'get_mil_default_params',
    'MIL_MODEL_REGISTRY',
    'MIL_DEFAULT_PARAMS'
] 