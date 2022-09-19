# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from mmcv import Config

from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.constants import IR, Task


def input_selector(options: Sequence,
                   prompt: str = 'Please select:',
                   index: bool = True,
                   default=''):

    options_str = options
    if index:
        options_str = [f'{i}: {opt}' for i, opt in enumerate(options)]

    if len(default) > 0:
        prompt += f' default({default}): '
    prompt = '\n'.join(options_str) + '\n' + prompt

    ret = input(prompt)
    if len(ret) == 0:
        ret = default
    if index and str(ret).isnumeric():
        ret = int(ret)
        assert ret >= 0 and ret < len(options), f'Index: {ret} out of range.'
        return options[ret]
    else:
        assert ret in options, f'Unknown option: {ret}.'
        return ret


def input_tuple(prompt: str = 'Input tuple:',
                num_inputs=None,
                default: str = ''):
    if len(default) > 0:
        prompt += f' default({default}):'
    ret = input(prompt)
    if len(ret) == 0:
        ret = default

    try:
        ret = eval(ret)
    except SyntaxError as e:
        print('Syntax Error. Please recheck input.')
        raise e

    assert isinstance(ret, Sequence)
    if num_inputs is not None:
        assert len(
            ret
        ) == num_inputs, f'Expect {num_inputs} value, get {len(ret)} value.'

    ret = tuple(ret)

    return ret


def input_bool(prompt: str = 'Input trigger:', default: str = ''):
    prompt += ' (y/n)'
    if len(default) > 0:
        prompt += f' default({default}):'

    ret = input(prompt)
    if len(ret) == 0:
        ret = default

    if ret.lower() in ['y', 'yes', 'true', '1']:
        ret = True
    elif ret.lower() in ['n', 'no', 'false', '0']:
        ret = False
    else:
        raise ValueError('Please enter yes or no.')

    return ret


def input_int(prompt: str = 'Input int:', default: str = '') -> int:
    if len(default) > 0:
        prompt += f' default({default}):'
    ret = input(prompt)
    if len(ret) == 0:
        ret = default

    if not ret.isnumeric():
        raise ValueError(f'Expect int, get {ret}')

    return int(ret)


def input_path(prompt: str = 'Input path:',
               check_exist: bool = True,
               default: str = '') -> str:
    if len(default) > 0:
        prompt += f' default({default}):'
    ret = input(prompt)
    if len(ret) == 0:
        ret = default

    ret = osp.expanduser(ret)
    if check_exist:
        assert osp.exists(ret), f'Path {ret} not exist.'

    return ret


def select_task(codebase):
    if codebase == 'mmcls':
        return Task.CLASSIFICATION.value
    elif codebase == 'mmdet':
        options = [
            Task.OBJECT_DETECTION.value, Task.INSTANCE_SEGMENTATION.value
        ]
        return input_selector(options, prompt='Please select task:')
    elif codebase == 'mmseg':
        return Task.SEGMENTATION.value
    elif codebase == 'mmocr':
        options = [Task.TEXT_DETECTION.value, Task.TEXT_RECOGNITION.value]
        return input_selector(options, prompt='Please select task:')
    elif codebase == 'mmedit':
        return Task.SUPER_RESOLUTION.value
    elif codebase == 'mmdet3d':
        return Task.VOXEL_DETECTION.value
    elif codebase == 'mmpose':
        return Task.POSE_DETECTION.value
    elif codebase == 'mmrotate':
        return Task.ROTATED_DETECTION.value
    else:
        raise NotImplementedError(f'Unsupported codebase: {codebase}.')


def select_ir(backend):
    if backend in [Backend.COREML.value, Backend.TORCHSCRIPT.value]:
        return IR.TORCHSCRIPT.value
    else:
        return IR.ONNX.value


def update_ir_config(cfg: Config, ir: str) -> None:
    if ir == IR.ONNX.value:
        cfg.merge_from_dict(
            dict(
                ir_config=dict(
                    type='onnx',
                    export_params=True,
                    keep_initializers_as_inputs=False,
                    opset_version=11,
                    save_file='end2end.onnx',
                    optimize=True)))
    elif ir == IR.TORCHSCRIPT.value:
        cfg.merge_from_dict(
            dict(ir_config=dict(type='torchscript', save_file='end2end.pt')))
    else:
        raise NotImplementedError(
            f'Update IR config failed. Unsupported IR: {ir}.')


def update_task(cfg: Config,
                task: str,
                is_static: bool = False,
                input_shape: Optional[Tuple[int]] = None) -> None:

    # update IO names in ir_config
    update_config = None
    if task == Task.CLASSIFICATION.value:
        update_config = dict(
            ir_config=dict(
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {
                        0: 'batch',
                        2: 'height',
                        3: 'width'
                    },
                    'output': {
                        0: 'batch'
                    }
                }),
            codebase_config=dict(type='mmcls', task='Classification'))
    elif task == Task.OBJECT_DETECTION.value:
        update_config = dict(
            ir_config=dict(
                input_names=['input'],
                output_names=['dets', 'labels'],
                dynamic_axes={
                    'input': {
                        0: 'batch',
                        2: 'height',
                        3: 'width'
                    },
                    'dets': {
                        0: 'batch',
                        1: 'num_dets',
                    },
                    'labels': {
                        0: 'batch',
                        1: 'num_dets',
                    },
                }),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
                model_type='end2end',
                post_processing=dict(
                    score_threshold=0.05,
                    confidence_threshold=0.005,  # for YOLOv3
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=5000,
                    keep_top_k=100,
                    background_label_id=-1,
                )))
    elif task == Task.INSTANCE_SEGMENTATION.value:
        update_config = dict(
            ir_config=dict(
                input_names=['input'],
                output_names=['dets', 'labels', 'masks'],
                dynamic_axes={
                    'input': {
                        0: 'batch',
                        2: 'height',
                        3: 'width'
                    },
                    'dets': {
                        0: 'batch',
                        1: 'num_dets',
                    },
                    'labels': {
                        0: 'batch',
                        1: 'num_dets',
                    },
                    'masks': {
                        0: 'batch',
                        1: 'num_dets',
                        2: 'height',
                        3: 'width'
                    },
                }),
            codebase_config=dict(
                type='mmdet',
                task='ObjectDetection',
                model_type='end2end',
                post_processing=dict(
                    score_threshold=0.05,
                    confidence_threshold=0.005,  # for YOLOv3
                    iou_threshold=0.5,
                    max_output_boxes_per_class=200,
                    pre_top_k=5000,
                    keep_top_k=100,
                    background_label_id=-1,
                    export_postprocess_mask=False)))
    elif task == Task.POSE_DETECTION.value:
        update_config = dict(
            ir_config=dict(
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {
                        0: 'batch',
                    },
                    'output': {
                        0: 'batch'
                    }
                }),
            codebase_config=dict(type='mmpose', task='PoseDetection'))
    elif task == Task.ROTATED_DETECTION.value:
        update_config = dict(
            ir_config=dict(
                input_names=['input'],
                output_names=['dets', 'labels'],
                dynamic_axes={
                    'input': {
                        0: 'batch',
                        2: 'height',
                        3: 'width'
                    },
                    'dets': {
                        0: 'batch',
                        1: 'num_dets',
                    },
                    'labels': {
                        0: 'batch',
                        1: 'num_dets',
                    },
                }),
            codebase_config=dict(
                type='mmrotate',
                task='RotatedDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.1,
                    pre_top_k=3000,
                    keep_top_k=2000,
                    max_output_boxes_per_class=2000)))
    elif task == Task.SEGMENTATION.value:
        update_config = dict(
            ir_config=dict(
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {
                        0: 'batch',
                        2: 'height',
                        3: 'width'
                    },
                    'output': {
                        0: 'batch',
                        2: 'height',
                        3: 'width'
                    },
                }),
            codebase_config=dict(type='mmseg', task='Segmentation'))
    elif task == Task.SUPER_RESOLUTION.value:
        update_config = dict(
            ir_config=dict(
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {
                        0: 'batch',
                        2: 'height',
                        3: 'width'
                    },
                    'output': {
                        0: 'batch',
                        2: 'height',
                        3: 'width'
                    }
                }),
            codebase_config=dict(type='mmedit', task='SuperResolution'))
    elif task == Task.TEXT_DETECTION.value:
        update_config = dict(
            ir_config=dict(
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {
                        0: 'batch',
                        2: 'height',
                        3: 'width'
                    },
                    'output': {
                        0: 'batch',
                        2: 'height',
                        3: 'width'
                    }
                }),
            codebase_config=dict(type='mmocr', task='TextDetection'))
    elif task == Task.TEXT_RECOGNITION.value:
        update_config = dict(
            ir_config=dict(
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {
                        0: 'batch',
                        3: 'width'
                    },
                    'output': {
                        0: 'batch',
                        1: 'seq_len',
                        2: 'num_classes'
                    }
                }),
            codebase_config=dict(type='mmocr', task='TextRecognition'))
    elif task == Task.VOXEL_DETECTION.value:
        update_config = dict(
            ir_config=dict(
                input_names=['voxels', 'num_points', 'coors'],
                output_names=['scores', 'bbox_preds', 'dir_scores'],
                dynamic_axes={
                    'voxels': {
                        0: 'voxels_num',
                    },
                    'num_points': {
                        0: 'voxels_num',
                    },
                    'coors': {
                        0: 'voxels_num',
                    }
                }),
            codebase_config=dict(
                type='mmdet3d', task='VoxelDetection', model_type='end2end'))
    else:
        raise NotImplementedError(
            f'Update task failed. Unsupported Task: {task}.')

    ir = cfg.ir_config.type
    if ir != IR.ONNX.value or is_static:
        update_config['ir_config'].pop('dynamic_axes')
    if is_static and len(input_shape) == 0:
        raise ValueError('Input shape should not be empty')
    if len(input_shape) == 0:
        input_shape = None
    update_config['ir_config']['input_shape'] = input_shape
    cfg.merge_from_dict(update_config)


def update_ascend(cfg: Config, **kwargs) -> None:
    raise NotImplementedError


def update_coreml(cfg: Config, **kwargs) -> None:
    raise NotImplementedError


def update_ncnn(cfg: Config, **kwargs) -> None:
    backend_config = dict(type='ncnn', use_vulkan=False)

    precision = input_selector(['FP32', 'INT8'],
                               'ncnn precision:',
                               default='FP32')
    backend_config['precision'] = precision
    cfg.merge_from_dict(dict(backend_config=backend_config))


def update_onnxruntime(cfg: Config, **kwargs) -> None:
    cfg.merge_from_dict(dict(backend_config=dict(type='onnxruntime')))


def update_openvino(cfg: Config, **kwargs) -> None:
    raise NotImplementedError


def update_pplnn(cfg: Config, **kwargs) -> None:
    raise NotImplementedError


def update_rknn(cfg: Config, **kwargs) -> None:
    raise NotImplementedError


def update_snpe(cfg: Config, **kwargs) -> None:
    cfg.merge_from_dict(dict(backend_config=dict(type='snpe')))


def update_tensorrt(cfg: Config,
                    is_static: bool,
                    input_shape: Optional[Dict[str, Tuple[int]]] = None,
                    **kwargs) -> None:
    backend_config = dict(type='tensorrt')
    calib_config = None
    fp16_mode = input_bool('Enable fp16 mode', default='n')
    int8_mode = input_bool('Enable int8 mode', default='n')
    if int8_mode:
        # calibration config
        calib_config = dict(create_calib=True, calib_file='calib_data.h5')
    max_workspace_size = input_int('Max workspace size', default='1073741824')

    # common config
    backend_config['common_config'] = dict(
        fp16_mode=fp16_mode,
        int8_mode=int8_mode,
        max_workspace_size=max_workspace_size)

    update_config = dict(backend_config=backend_config)
    if calib_config is not None:
        update_config['calib_config'] = calib_config
    cfg.merge_from_dict(update_config)

    # input shape
    codebase_config = cfg.codebase_config
    codebase = codebase_config.type

    trt_input_shapes = dict()

    if codebase in [
            'mmcls', 'mmdet', 'mmedit', 'mmocr', 'mmpose', 'mmrotate', 'mmseg'
    ]:
        input_name = cfg.ir_config.input_names[0]
        if is_static:
            min_shape = opt_shape = max_shape = [
                1, 3, input_shape[1], input_shape[0]
            ]
            trt_input_shapes[input_name] = dict(
                min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape)
        else:
            from mmdeploy.apis import build_task_processor
            model_cfg_path = input_path(
                'Please provide model config:', check_exist=True, default='')
            min_batch, max_batch = input_tuple(
                'batch size range:', num_inputs=2, default='(1,1)')
            model_cfg = Config.fromfile(model_cfg_path)
            task = build_task_processor(model_cfg, cfg, 'cuda')

            width_range = input_tuple('image width range:', num_inputs=2)
            height_range = input_tuple('image height range:', num_inputs=2)
            assert width_range[0] <= width_range[1]
            assert height_range[0] <= height_range[1]

            dummy_inputs = [
                np.zeros((height_range[i % 2], width_range[i // 2], 3),
                         dtype=np.uint8) for i in range(4)
            ]
            tensors = [
                task.create_input(dummy_input)[1]
                for dummy_input in dummy_inputs
            ]

            if not isinstance(tensors[0], torch.Tensor):
                tensors = [t[0] for t in tensors]

            min_width = min(t.shape[-1] for t in tensors)
            max_width = max(t.shape[-1] for t in tensors)
            min_height = min(t.shape[-2] for t in tensors)
            max_height = max(t.shape[-2] for t in tensors)
            trt_input_shapes[input_name] = dict(
                min_shape=[min_batch, 3, min_height, min_width],
                opt_shape=[min_batch, 3, min_height, min_width],
                max_shape=[max_batch, 3, max_height, max_width],
            )
    else:
        # mmdet3d
        input_names = cfg.ir_config.input_names
        for name in input_names:
            min_shape = input_tuple(f'Min shape of input <{name}>:')
            opt_shape = input_tuple(f'Opt shape of input <{name}>:')
            max_shape = input_tuple(f'Max shape of input <{name}>:')
            trt_input_shapes[name] = dict(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            )

    backend_config['model_inputs'] = [dict(input_shapes=trt_input_shapes)]
    cfg.merge_from_dict(update_config)


def update_torchscript(cfg: Config, **kwargs) -> None:
    cfg.merge_from_dict(dict(backend_config=dict(type='torchscript')))


def update_backend(cfg: Config,
                   backend: str,
                   is_static: bool,
                   input_shape: Optional[Tuple[int]] = None) -> None:
    assert 'codebase_config' in cfg, 'Please update task info first.'
    codebase_config = cfg.codebase_config
    assert 'type' in codebase_config and 'task' in codebase_config, \
        'type and task not found in codebase config.'
    codebase = codebase_config.type
    task = codebase_config.task

    assert codebase in [co.value
                        for co in Codebase], f'Unknown codebase: {codebase}'
    assert task in [ta.value for ta in Task], f'Unknown task: {task}'

    backend_options = [be.value for be in Backend]
    backend_exclude = ['pytorch', 'default']
    backend_options = list(
        filter(lambda x: x not in backend_exclude, backend_options))
    if backend in backend_options:
        backend_func = eval(f'update_{backend}')
        backend_func(cfg, is_static=is_static, input_shape=input_shape)
    else:
        raise NotImplementedError(f'Unsupported backend: {backend}.')


def main():
    backend_options = [be.value for be in Backend]
    backend_exclude = ['pytorch', 'default']
    backend_options = list(
        filter(lambda x: x not in backend_exclude, backend_options))
    backend = input_selector(
        backend_options, prompt='please choice the backend:')
    codebase_options = [cb.value for cb in Codebase]
    codebase = input_selector(
        codebase_options, prompt='please choice the codebase:')

    task = select_task(codebase)
    ir = select_ir(backend)

    is_static = False
    if codebase != Codebase.MMDET3D.value:
        is_static = input_bool('Static input shape?', default='n')
        input_shape = input_tuple(
            'Input tensor size (width, height):', default='tuple()')

    cfg = Config()
    update_ir_config(cfg, ir)
    update_task(cfg, task, is_static=is_static, input_shape=input_shape)
    update_backend(cfg, backend, is_static=is_static, input_shape=input_shape)

    print('\ndump config:')
    print(cfg.pretty_text)

    print('\n')
    save_path = input_path('save config path:', check_exist=False, default='')

    if len(save_path) > 0:
        cfg.dump(save_path)
        print(f'Save config to {save_path}.')
    else:
        print('Invalid save path.')


if __name__ == '__main__':
    main()
