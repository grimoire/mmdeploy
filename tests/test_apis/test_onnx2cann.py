# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import pytest
import torch
import torch.nn as nn

from mmdeploy.backend.cann import get_atc_options_from_cfg
from mmdeploy.utils import Backend
from mmdeploy.utils.test import backend_checker

onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx').name
test_img = torch.rand([1, 3, 8, 8])


@pytest.mark.skip(reason='This a not test class but a utility class.')
class TestModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5


test_model = TestModel().eval()


def generate_onnx_file(model):
    with torch.no_grad():
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'output': {
                0: 'batch'
            }
        }
        torch.onnx.export(
            model,
            test_img,
            onnx_file,
            output_names=['output'],
            input_names=['input'],
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11,
            dynamic_axes=dynamic_axes)
        assert osp.exists(onnx_file)


@backend_checker(Backend.CANN)
def test_onnx2cann():
    from mmcv import Config

    from mmdeploy.apis.cann import from_onnx
    model = test_model
    generate_onnx_file(model)

    work_dir, _ = osp.split(onnx_file)
    cfg = Config(
        dict(
            backend_config=dict(model_inputs=[
                dict(
                    input_shapes=dict(input=[-1, 3, 8, 8]),
                    dynamic_batch=[1, 2])
            ])))
    opts = get_atc_options_from_cfg(cfg)
    file_name = osp.splitext(onnx_file)[0]
    om_file = osp.join(work_dir, file_name + '.om')
    from_onnx(onnx_file, osp.join(work_dir, file_name), **opts)
    assert osp.exists(work_dir)
    assert osp.exists(om_file)
