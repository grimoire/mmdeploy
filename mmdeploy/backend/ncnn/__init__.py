# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import NCNNManager, NCNNParam
from .onnx2ncnn import from_onnx

_BackendManager = NCNNManager

is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['NCNNParam', 'NCNNManager', 'from_onnx']
