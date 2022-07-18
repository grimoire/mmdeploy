# Copyright (c) OpenMMLab. All rights reserved.
import importlib


def is_available():
    """Ascend available.

    Returns:
        bool: Always True.
    """
    acl = importlib.util.find_spec('acl') is not None

    return acl is not None


__all__ = []

if is_available():
    try:
        from .utils import get_atc_options_from_cfg
        from .wrapper import CANNWrapper

        __all__ += ['CANNWrapper', 'get_atc_options_from_cfg']
    except Exception:
        pass
