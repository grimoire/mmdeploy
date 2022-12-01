# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
from .backend_utils import TorchScriptUtils
from .init_plugins import get_ops_path, ops_available


def is_available():
    """Torchscript available.

    Returns:
        bool: Always True.
    """
    return True


__all__ = ['get_ops_path', 'ops_available', 'TorchScriptUtils']

if is_available():
    from .wrapper import TorchscriptWrapper

    __all__ += ['TorchscriptWrapper']
