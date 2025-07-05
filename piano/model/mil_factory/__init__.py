"""
MIL Factory - Multiple Instance Learning Models Factory

This module provides a unified interface for creating MIL baseline models.
"""

import torch
import torch.nn as nn

# ============================================================================
# Part 1: Import all MIL models
# ============================================================================

from .abmil.abmil import ABMIL, GatedABMIL
from .clam.clam import CLAM_SB, CLAM_MB
from .transmil.transmil import TransMIL
from .dgrmil.dgrmil import DGRMIL
from .dtfdmil.dtfdmil import DTFDMIL
from .dsmil.dsmil import DSMIL
from .ilramil.ilramil import ILRAMIL
from .wikg.wikg import WiKG
from .s4mil.s4mil import S4MIL
from .amdmil.amdmil import AMD_MIL
from .rrtmil.rrtmil import RRT_MIL
from .pooling.mil import MeanPool, MaxPool
from .retmil.retmil import RetMIL
from .mamba_2d.mamba_2d import MambaMIL_2D
from .m4.m4 import M4

# ============================================================================
# Part 2: Model Registry
# ============================================================================

MIL_MODEL_REGISTRY = {
    'abmil': ABMIL,
    'gated_abmil': GatedABMIL,
    'clam_sb': CLAM_SB,
    'clam_mb': CLAM_MB,
    'transmil': TransMIL,
    'dgrmil': DGRMIL,
    'dtfdmil': DTFDMIL,
    'dsmil': DSMIL,
    'ilramil': ILRAMIL,
    'wikg': WiKG,
    's4mil': S4MIL,
    'amdmil': AMD_MIL,
    'rrtmil': RRT_MIL,
    'mean_pool': MeanPool,
    'max_pool': MaxPool,
    'retmil': RetMIL,
    '2dmamba': MambaMIL_2D,
    'm4': M4,
}

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
    'rrtmil': {'dim_in': 1024, 'dim_hidden': 512, 'num_classes': 2, 'dropout': 0.25, 'survival': False},
    's4mil': {'dim_in': 1024, 'num_classes': 2, 'dropout': 0.25, 'act': 'gelu', 'survival': False},
    'transmil': {'dim_in': 1024, 'dim_hidden': 512, 'num_classes': 2, 'num_layers': 2, 'num_heads': 8, 'dropout': 0.25, 'survival': False},
    'wikg': {'dim_in': 384, 'dim_hidden': None, 'num_classes': 2, 'topk': 6, 'agg_type': 'bi-interaction', 'dropout': 0.3, 'pool': 'attn', 'survival': False},
    'mean_pool': {'dim_in': 1024, 'num_classes': 2, 'survival': False},
    'max_pool': {'dim_in': 1024, 'num_classes': 2, 'survival': False},
    'retmil': {'dim_in': 1024, 'num_heads': 8, 'window_size': 256, 'stride': 256, 'num_classes': 2, 'survival': False},
    'amdmil': {'dim_in': 1024, 'embed_dim': 512, 'num_classes': 10, 'agent_num': 256, 'survival': False}, 
    '2dmamba': {'dim_in': 1024, 'drop_out': 0.25, 'num_classes': 2, 'survival': False, 'pos_emb_type': None},
    'm4': {'dim_in': 1024, 'experts_out': 512, 'towers_out': 2, 'num_classes': 2, 'towers_hidden': 128, 'tasks': 1, 'num_expert': 4, 'survival': False},
}

# ============================================================================
# Part 4: Model Creation Function
# ============================================================================

def create_mil_model(model_name, **kwargs):
    """
    Create MIL model
    
    Args:
        model_name (str): Model name
        **kwargs: Model parameters that will override default parameters
        
    Returns:
        torch.nn.Module: Created model instance
        
    Raises:
        ValueError: If mil_name is not recognized
    """
    if model_name not in MIL_MODEL_REGISTRY:
        available_models = list(MIL_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
    
    model_class = MIL_MODEL_REGISTRY[model_name]
    
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
    return list(MIL_MODEL_REGISTRY.keys())


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
    if mil_name not in MIL_MODEL_REGISTRY:
        available_models = list(MIL_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {mil_name}. Available models: {available_models}")
    
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