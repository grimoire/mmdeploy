# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Sequence

from ..base import BACKEND_UTILS, BaseBackendUtils


@BACKEND_UTILS.register('coreml')
class CoreMLUtils(BaseBackendUtils):

    def build_wrapper(backend_files: Sequence[str],
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
        from .wrapper import CoreMLWrapper
        return CoreMLWrapper(model_file=backend_files[0])
