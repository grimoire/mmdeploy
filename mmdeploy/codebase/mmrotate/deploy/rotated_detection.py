# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine import Config
from mmengine.dataset import pseudo_collate
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Codebase, Task
from mmdeploy.utils.config_utils import get_input_shape, is_dynamic_shape

MMROTATE_TASK = Registry('mmrotate_tasks')


@CODEBASE.register_module(Codebase.MMROTATE.value)
class MMRotate(MMCodebase):
    """MMRotate codebase class."""

    task_registry = MMROTATE_TASK

    @classmethod
    def register_deploy_modules(cls):
        import mmdeploy.codebase.mmrotate.models  # noqa: F401
        import mmdeploy.codebase.mmrotate.structures  # noqa: F401
        from mmdeploy.codebase.mmdet.deploy.object_detection import MMDetection
        MMDetection.register_deploy_modules()

    @classmethod
    def register_all_modules(cls):
        from mmdet.utils.setup_env import \
            register_all_modules as register_all_modules_mmdet
        from mmrotate.utils.setup_env import register_all_modules

        cls.register_deploy_modules()
        register_all_modules(True)
        register_all_modules_mmdet(False)


def replace_RResize(pipelines):
    """Rename RResize to Resize.

    args:
        pipelines (list[dict]): Data pipeline configs.

    Returns:
        list: The new pipeline list with all RResize renamed to
            Resize.
    """
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_RResize(pipeline['transforms'])
        elif pipeline.type == 'RResize':
            pipelines[i].type = 'Resize'
            if 'keep_ratio' not in pipelines[i]:
                pipelines[i]['keep_ratio'] = True  # default value
    return pipelines


def process_model_config(model_cfg: Config,
                         imgs: Union[Sequence[str], Sequence[np.ndarray]],
                         input_shape: Optional[Sequence[int]] = None):
    """Process the model config.

    Args:
        model_cfg (Config): The model config.
        imgs (Sequence[str] | Sequence[np.ndarray]): Input image(s), accepted
            data type are List[str], List[np.ndarray].
        input_shape (list[int]): A list of two integer in (width, height)
            format specifying input shape. Default: None.

    Returns:
        Config: the model config after processing.
    """
    cfg = model_cfg.copy()

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.test_pipeline[0].type = 'LoadImageFromNDArray'

    pipeline = cfg.test_pipeline

    for i, transform in enumerate(pipeline):
        # for static exporting
        if input_shape is not None and transform.type == 'Resize':
            pipeline[i].keep_ratio = False
            pipeline[i].scale = tuple(input_shape)

    pipeline = [
        transform for transform in pipeline
        if transform.type != 'LoadAnnotations'
    ]
    cfg.test_pipeline = pipeline
    return cfg


def _get_dataset_metainfo(model_cfg: Config):
    """Get metainfo of dataset.

    Args:
        model_cfg Config: Input model Config object.

    Returns:
        list[str]: A list of string specifying names of different class.
    """
    from mmdet import datasets  # noqa
    from mmdet.registry import DATASETS

    module_dict = DATASETS.module_dict

    for dataloader_name in [
            'test_dataloader', 'val_dataloader', 'train_dataloader'
    ]:
        if dataloader_name not in model_cfg:
            continue
        dataloader_cfg = model_cfg[dataloader_name]
        dataset_cfg = dataloader_cfg.dataset
        dataset_cls = module_dict.get(dataset_cfg.type, None)
        if dataset_cls is None:
            continue
        if hasattr(dataset_cls, '_load_metainfo') and isinstance(
                dataset_cls._load_metainfo, Callable):
            meta = dataset_cls._load_metainfo(
                dataset_cfg.get('metainfo', None))
            if meta is not None:
                return meta
        if hasattr(dataset_cls, 'METAINFO'):
            return dataset_cls.METAINFO

    return None


@MMROTATE_TASK.register_module(Task.ROTATED_DETECTION.value)
class RotatedDetection(BaseTask):
    """Rotated detection task class.

    Args:
        model_cfg (Config): Loaded model Config object..
        deploy_cfg (Config): Loaded deployment Config object.
        device (str): A string represents device type.
        experiment_name (str): The experiment name. Default to
            `RotatedDetection`.
    """

    def __init__(self,
                 model_cfg: Config,
                 deploy_cfg: Config,
                 device: str,
                 experiment_name: str = 'RotatedDetection'):
        super().__init__(model_cfg, deploy_cfg, device, experiment_name)

    def build_backend_model(self,
                            model_files: Optional[str] = None,
                            **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .rotated_detection_model import build_rotated_detection_model
        data_preprocessor = copy.deepcopy(
            self.model_cfg.model.get('data_preprocessor', {}))
        data_preprocessor.setdefault('type', 'mmdet.DetDataPreprocessor')

        model = build_rotated_detection_model(
            model_files,
            self.deploy_cfg,
            device=self.device,
            data_preprocessor=data_preprocessor)
        model = model.to(self.device)
        return model.eval()

    def create_input(
        self,
        imgs: Union[str, np.ndarray],
        input_shape: Sequence[int] = None,
        data_preprocessor: Optional[BaseDataPreprocessor] = None
    ) -> Tuple[Dict, torch.Tensor]:
        """Create input for rotated object detection.

        Args:
            imgs (str | np.ndarray): Input image(s), accepted data type are
            `str`, `np.ndarray`.
            input_shape (list[int]): A list of two integer in (width, height)
                format specifying input shape. Defaults to `None`.

        Returns:
            tuple: (data, img), meta information for the input image and input.
        """
        from mmcv.transforms import Compose
        if isinstance(imgs, (list, tuple)):
            if not isinstance(imgs[0], (np.ndarray, str)):
                raise AssertionError('imgs must be strings or numpy arrays')
        elif isinstance(imgs, (np.ndarray, str)):
            imgs = [imgs]
        else:
            raise AssertionError('imgs must be strings or numpy arrays')

        dynamic_flag = is_dynamic_shape(self.deploy_cfg)
        cfg = process_model_config(self.model_cfg, imgs, input_shape)

        pipeline = cfg.test_pipeline
        if not dynamic_flag:
            transform = pipeline[1]
            if 'transforms' in transform:
                transform_list = transform['transforms']
                for i, step in enumerate(transform_list):
                    if step['type'] == 'Pad' and 'pad_to_square' in step \
                       and step['pad_to_square']:
                        transform_list.pop(i)
                        break
        test_pipeline = Compose(pipeline)

        data = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                # TODO: remove img_id.
                data_ = dict(img=img, img_id=0)
            else:
                # TODO: remove img_id.
                data_ = dict(img_path=img, img_id=0)
            # build the data pipeline
            data_ = test_pipeline(data_)
            data.append(data_)

        data = pseudo_collate(data)
        if data_preprocessor is not None:
            data = data_preprocessor(data, False)
            return data, data['inputs']
        else:
            return data, BaseTask.get_tensor_from_input(data)

    @staticmethod
    def get_partition_cfg(partition_type: str) -> Dict:
        """Get a certain partition config.

        Args:
            partition_type (str): A string specifying partition type.

        Returns:
            dict: A dictionary of partition config.
        """
        raise NotImplementedError('Not supported yet.')

    def get_preprocess(self, *args, **kwargs) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        input_shape = get_input_shape(self.deploy_cfg)
        model_cfg = process_model_config(self.model_cfg, [''], input_shape)
        # rename sdk RResize -> Resize
        model_cfg.data.test.pipeline = replace_RResize(
            model_cfg.data.test.pipeline)
        preprocess = model_cfg.data.test.pipeline
        return preprocess

    def get_postprocess(self, *args, **kwargs) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Composed of the postprocess information.
        """
        postprocess = self.model_cfg.model.test_cfg
        return postprocess

    def get_model_name(self, *args, **kwargs) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'type' in self.model_cfg.model, 'model config contains no type'
        name = self.model_cfg.model.type.lower()
        return name

    def get_visualizer(self, name: str, save_dir: str):
        visualizer = super().get_visualizer(name, save_dir)
        metainfo = _get_dataset_metainfo(self.model_cfg)
        if metainfo is not None:
            visualizer.dataset_meta = metainfo
        return visualizer

    def visualize(self, image: Union[str, np.ndarray], result: list,
                  output_file: str, *args, **kwargs):
        """Visualize predictions of a model.

        Args:
            image (str | np.ndarray): Input image to draw predictions on.
            result (list): A list of predictions.
            output_file (str): Output file to save drawn image.
            window_name (str): The name of visualization window. Defaults to
                an empty string.
            show_result (bool): Whether to show result in windows, defaults
                to `False`.
            draw_gt (bool): Whether to show ground truth in windows, defaults
                to `False`.
        """
        from mmrotate.structures.bbox import RotatedBoxes
        pred_instances = result._pred_instances
        pred_instances.scores = pred_instances.scores.detach().cpu()
        pred_instances.bboxes = pred_instances.bboxes.detach().cpu()
        pred_instances.labels = pred_instances.labels.detach().cpu()
        if isinstance(pred_instances.bboxes, torch.Tensor):
            pred_instances.bboxes = RotatedBoxes(
                pred_instances.bboxes, clone=True)

        return super().visualize(image, result.cpu(), output_file, *args,
                                 **kwargs)
