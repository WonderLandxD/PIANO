from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmselfsup.models import build_algorithm
# from mmengine.registry import MODELS as MMENGINE_MODELS
# from mmengine.registry import Registry

# MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['mmselfsup.models'])

from typing import Optional, Union
import torch
import torch.nn as nn


def init_model(config: Union[str, Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               options: Optional[dict] = None) -> nn.Module:
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): The device where the anchors will be put on.
            Defaults to cuda:0.
        options (dict): Options to override some settings in the used config.

    Returns:
        nn.Module: The initialized model.
    """
    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    if options is not None:
        config.merge_from_dict(options)
    init_default_scope(config.get('default_scope', 'mmselfsup'))

    config.model.pretrained = None
    config.model.setdefault('data_preprocessor',
                            config.get('data_preprocessor', None))
    model = build_algorithm(config.model)
    if checkpoint is not None:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    cfg = Config.fromfile('/mnt/sdb/ljw/PIANO-Custom/beph_backbone.py')

    model = init_model(cfg, checkpoint='/mnt/sdb/ljw/PIANO-Custom/BEPH_backbone.pth', device='cpu')

    input = torch.randn(1, 3, 224, 224)
    output = model.extract_feat(input)[0] # .cpu().detach().numpy()
    # tensor_output = torch.from_numpy(output)
    print(output.shape)

    # print(model)