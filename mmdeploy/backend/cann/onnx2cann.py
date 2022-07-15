# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from subprocess import PIPE, run
from typing import Dict, Optional, Sequence, Tuple, Union

import onnx

from mmdeploy.utils import get_root_logger

logger = get_root_logger()


def from_onnx(onnx_model: Union[str, onnx.ModelProto],
              output_file_prefix: str,
              input_shapes: Dict[str, Sequence[int]],
              dynamic_batch: Optional[Sequence[int]] = None,
              dynamic_hw: Optional[Sequence[Tuple[int, int]]] = None,
              soc_name: Optional[str] = None,
              extra_options: Optional[str] = None) -> None:
    """Convert ONNX to CANN model.

    Args:
        onnx_model (Union[str, onnx.ModelProto]): The onnx model or it's path.
        output_file_prefix (str): The path to the directory for saving
            the results.
        input_shapes (Dict[str, Sequence[int]]): Shape of input data.
        dynamic_batch (Optional[Sequence[int]], optional): Set dynamic batch
            size. Defaults to None.
        dynamic_hw (Optional[Sequence[Tuple[int, int]]], optional): Set dynamic
            height and width size. Defaults to None.
        soc_name (Optional[str], optional): The soc name. Defaults to None.
        extra_options (Optional[str], optional): Other flag to feed to atc.
            Defaults to None.
    """
    framework = 5  # onnx framework == 5

    # set model path
    if isinstance(onnx_model, str):
        onnx_path = onnx_model
    else:
        onnx_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
        onnx.save(onnx_model, onnx_path)

    # set soc_version
    if soc_name is None:
        try:
            import acl
            soc_name = acl.get_soc_name()
        except Exception:
            soc_name = 'Ascend310'

    # set input shape
    input_shape_str = [
        f'{name}:{",".join([str(s) for s in shape])}'
        for name, shape in input_shapes.items()
    ]
    input_shape_str = ';'.join(input_shape_str)

    # required args
    cann_args = f'--model "{onnx_path}" '
    cann_args += f'--output "{output_file_prefix}" '
    cann_args += f'--framework {framework} '
    cann_args += f'--soc_version "{soc_name}" '
    cann_args += f'--input_shape "{input_shape_str}" '

    # optional args
    if dynamic_batch is not None:
        dynamic_batch_str = ','.join([str(b) for b in dynamic_batch])
        cann_args += f'--dynamic_batch_size "{dynamic_batch_str}" '

    if dynamic_hw is not None:
        dynamic_hw_str = ';'.join(
            [','.join([str(i) for i in hw]) for hw in dynamic_hw])
        cann_args += f'--dynamic_hw_size "{dynamic_hw_str}" '

    if extra_options is not None:
        cann_args += extra_options

    command = f'atc {cann_args}'
    logger.info(f'Convert with command: {command}')
    atc_output = run(command, stdout=PIPE, stderr=PIPE, shell=True, check=True)

    logger.info(atc_output.stdout.decode())
    logger.debug(atc_output.stderr.decode())

    logger.info(f'Successfully exported CANN model: {output_file_prefix}.om')
