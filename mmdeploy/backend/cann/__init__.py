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
        from .wrapper import CANNWrapper

        __all__ += ['CANNWrapper']
    except Exception:
        pass
