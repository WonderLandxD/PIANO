__version__ = "1.1.2"

from .ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from .modules.mamba_simple import Mamba
from .models.mixer_seq_simple import MambaLMHeadModel
from .modules.srmamba import SRMamba
from .modules.bimamba import BiMamba