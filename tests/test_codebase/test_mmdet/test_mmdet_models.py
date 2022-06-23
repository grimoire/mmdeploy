# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
from typing import Dict, List

import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.test import WrapModel, check_backend, get_rewrite_outputs

import_codebase(Codebase.MMDET)


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False


def convert_to_list(rewrite_output: Dict, output_names: List[str]) -> List:
    """Converts output from a dictionary to a list.

    The new list will contain only those output values, whose names are in list
    'output_names'.
    """
    outputs = [
        value for name, value in rewrite_output.items() if name in output_names
    ]
    return outputs


def get_anchor_head_model():
    """AnchorHead Config."""
    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    from mmdet.models.dense_heads import AnchorHead
    model = AnchorHead(num_classes=4, in_channels=1, test_cfg=test_cfg)
    model.requires_grad_(False)

    return model


def get_ssd_head_model():
    """SSDHead Config."""
    test_cfg = mmcv.Config(
        dict(
            nms_pre=1000,
            nms=dict(type='nms', iou_threshold=0.45),
            min_bbox_size=0,
            score_thr=0.02,
            max_per_img=200))

    from mmdet.models import SSDHead
    model = SSDHead(
        in_channels=(96, 1280, 512, 256, 256, 128),
        num_classes=4,
        use_depthwise=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='Normal', layer='Conv2d', std=0.001),
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            strides=[16, 32, 64, 107, 160, 320],
            ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
            min_sizes=[48, 100, 150, 202, 253, 304],
            max_sizes=[100, 150, 202, 253, 304, 320]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        test_cfg=test_cfg)

    model.requires_grad_(False)

    return model


def get_fcos_head_model():
    """FCOS Head Config."""
    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    from mmdet.models.dense_heads import FCOSHead
    model = FCOSHead(num_classes=4, in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


def get_focus_backbone_model():
    """Backbone Focus Config."""
    from mmdet.models.backbones.csp_darknet import Focus
    model = Focus(3, 32)

    model.requires_grad_(False)
    return model


def get_l2norm_forward_model():
    """L2Norm Neck Config."""
    from mmdet.models.necks.ssd_neck import L2Norm
    model = L2Norm(16)
    torch.nn.init.uniform_(model.weight)

    model.requires_grad_(False)
    return model


def get_rpn_head_model():
    """RPN Head Config."""
    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            nms_pre=0,
            max_per_img=100,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0))
    from mmdet.models.dense_heads import RPNHead
    model = RPNHead(in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


def get_reppoints_head_model():
    """Reppoints Head Config."""
    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    from mmdet.models.dense_heads import RepPointsHead
    model = RepPointsHead(num_classes=4, in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


def get_single_roi_extractor():
    """SingleRoIExtractor Config."""
    from mmdet.models.roi_heads import SingleRoIExtractor
    roi_layer = dict(type='RoIAlign', output_size=7, sampling_ratio=2)
    out_channels = 1
    featmap_strides = [4, 8, 16, 32]
    model = SingleRoIExtractor(roi_layer, out_channels, featmap_strides).eval()

    return model


def get_gfl_head_model():
    test_cfg = mmcv.Config(
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100))
    anchor_generator = dict(
        type='AnchorGenerator',
        scales_per_octave=1,
        octave_base_scale=8,
        ratios=[1.0],
        strides=[8, 16, 32, 64, 128])
    from mmdet.models.dense_heads import GFLHead
    model = GFLHead(
        num_classes=3,
        in_channels=256,
        reg_max=3,
        test_cfg=test_cfg,
        anchor_generator=anchor_generator)
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME, Backend.NCNN])
def test_focus_forward(backend_type):
    check_backend(backend_type)
    focus_model = get_focus_backbone_model()
    focus_model.cpu().eval()
    s = 128
    seed_everything(1234)
    x = torch.rand(1, 3, s, s)
    model_outputs = [focus_model.forward(x)]
    wrapped_model = WrapModel(focus_model, 'forward')
    rewrite_inputs = {
        'x': x,
    }
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(input_shape=None)))
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)
    for model_output, rewrite_output in zip(model_outputs[0],
                                            rewrite_outputs[0]):
        model_output = model_output.squeeze()
        rewrite_output = rewrite_output.squeeze()
        torch.testing.assert_allclose(
            model_output, rewrite_output, rtol=1e-03, atol=1e-05)
