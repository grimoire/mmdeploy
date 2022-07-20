# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict


def get_atc_options_from_cfg(cfg, index: int = 0) -> Dict:
    """Read act options from deploy config.

    Args:
        cfg (Config): The deploy config.
        index (int): The atc config index, default to 0.
    """
    from mmdeploy.utils import get_backend_config
    backend_cfg = get_backend_config(cfg)
    model_inputs = backend_cfg.get('model_inputs', [])
    assert len(
        model_inputs) > index, f'Can not find options for model {index}.'
    options = model_inputs[index]
    return options
