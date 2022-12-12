# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Optional, Sequence

from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('openvino')
class OpenVINOManager(BaseBackendManager):

    @classmethod
    def build_wrapper(cls,
                      backend_files: Sequence[str],
                      device: str = 'cpu',
                      input_names: Optional[Sequence[str]] = None,
                      output_names: Optional[Sequence[str]] = None,
                      deploy_cfg: Optional[Any] = None,
                      **kwargs):
        """Build the wrapper for the backend model.

        Args:
            backend_files (Sequence[str]): Backend files.
            device (str, optional): The device info. Defaults to 'cpu'.
            input_names (Optional[Sequence[str]], optional): input names.
                Defaults to None.
            output_names (Optional[Sequence[str]], optional): output names.
                Defaults to None.
            deploy_cfg (Optional[Any], optional): The deploy config. Defaults
                to None.
        """
        from .wrapper import OpenVINOWrapper
        return OpenVINOWrapper(
            ir_model_file=backend_files[0], output_names=output_names)

    @classmethod
    def to_backend(cls,
                   ir_files: Sequence[str],
                   deploy_cfg: Any,
                   work_dir: str,
                   log_level: int = 20,
                   device: str = 'cpu',
                   **kwargs) -> Sequence[str]:
        """Convert intermediate representation to given backend.

        Args:
            ir_files (Sequence[str]): The intermediate representation files.
            deploy_cfg (Any): The deploy config.
            work_dir (str): The work directory, backend files and logs should
                be save in this directory.
            log_level (int, optional): The log level. Defaults to logging.INFO.
            device (str, optional): The device type. Defaults to 'cpu'.

        Returns:
            Seqeuence[str]: Backend files.
        """
        from . import is_available
        assert is_available(), \
            'OpenVINO is not available, please install OpenVINO first.'

        from mmdeploy.apis.openvino import (get_input_info_from_cfg,
                                            get_mo_options_from_cfg,
                                            get_output_model_file)
        from mmdeploy.utils import get_ir_config
        from .onnx2openvino import from_onnx

        openvino_files = []
        for onnx_path in ir_files:
            model_xml_path = get_output_model_file(onnx_path, work_dir)
            input_info = get_input_info_from_cfg(deploy_cfg)
            output_names = get_ir_config(deploy_cfg).output_names
            mo_options = get_mo_options_from_cfg(deploy_cfg)
            from_onnx(onnx_path, work_dir, input_info, output_names,
                      mo_options)
            openvino_files.append(model_xml_path)

        return openvino_files
