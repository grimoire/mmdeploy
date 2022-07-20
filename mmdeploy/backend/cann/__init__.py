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
    from mmdeploy.utils import get_root_logger

    # require atc
    try:
        from .utils import get_atc_options_from_cfg

        __all__ += ['get_atc_options_from_cfg']
    except ImportError:
        get_root_logger().warning('Import atc tools failed.')

    # require pytorch
    try:
        from .wrapper import CANNWrapper
        __all__ += ['CANNWrapper']
    except Exception:
        get_root_logger().warning('Import CANNWrapper failed.')
