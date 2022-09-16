# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmcv import Config

from mmdeploy.utils import Backend, Codebase
from mmdeploy.utils.constants import IR, Task


def selector(options: Sequence,
             prompt: str = 'Please select:',
             index: bool = True,
             default=''):

    options_str = options
    if index:
        options_str = [f'{i}: {opt}' for i, opt in enumerate(options)]

    if len(default) > 0:
        prompt += f' default({default})'
    prompt = prompt + '\n' + '\n'.join(options_str) + '\n'

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


def select_task(codebase):
    if codebase == 'mmcls':
        return Task.CLASSIFICATION.value
    elif codebase == 'mmdet':
        options = [
            Task.OBJECT_DETECTION.value, Task.INSTANCE_SEGMENTATION.value
        ]
        return selector(options, prompt='Please select task:')
    elif codebase == 'mmseg':
        return Task.SEGMENTATION.value
    elif codebase == 'mmocr':
        options = [Task.TEXT_DETECTION.value, Task.TEXT_RECOGNITION.value]
        return selector(options, prompt='Please select task:')
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


def update_task(cfg: Config, task: str) -> None:

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
    if ir != IR.ONNX.value:
        update_config['ir_config'].pop('dynamic_axes')
    cfg.merge_from_dict(update_config)


def update_onnxruntime(cfg: Config) -> None:
    pass


def update_backend(cfg: Config, backend: str) -> None:
    assert 'codebase_config' in cfg, 'Please update task info first.'
    codebase_config = cfg.codebase_cfg
    assert 'type' in codebase_config and 'task' in codebase_config, \
        'type and task not found in codebase config.'
    codebase = codebase_config.type
    task = codebase_config.task

    assert codebase in [co.value
                        for co in Codebase], f'Unknown codebase: {codebase}'
    assert task in [ta.value for ta in Task], f'Unknown task: {task}'

    if backend == Backend.ONNXRUNTIME.value:
        update_onnxruntime(cfg)
    else:
        raise NotImplementedError(f'Unsupported backend: {backend}.')


def main():
    backend_options = [be.value for be in Backend]
    backend_exclude = ['pytorch', 'default']
    backend_options = list(
        filter(lambda x: x not in backend_exclude, backend_options))
    backend = selector(backend_options, prompt='please choice the backend:')
    print('select backend:', backend)

    codebase_options = [cb.value for cb in Codebase]
    codebase = selector(codebase_options, prompt='please choice the codebase:')
    print('select codebase:', codebase)

    task = select_task(codebase)
    ir = select_ir(backend)

    cfg = Config()
    update_ir_config(cfg, ir)
    update_task(cfg, task)

    print('dump config:')
    print(cfg.pretty_text)


if __name__ == '__main__':
    main()
