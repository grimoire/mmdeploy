# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.cann import is_available

__all__ = ['is_available']

if is_available():
    from mmdeploy.backend.cann.onnx2cann import from_onnx as _from_onnx
    from ..core import PIPELINE_MANAGER

    from_onnx = PIPELINE_MANAGER.register_pipeline()(_from_onnx)

    __all__ += ['from_onnx']
