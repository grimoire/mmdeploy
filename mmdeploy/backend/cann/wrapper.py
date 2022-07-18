# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

import acl
import torch
import torch_npu

from mmdeploy.utils import Backend, get_root_logger
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper

ACL_MEMCPY_DEVICE_TO_DEVICE = 3

ACL_TORCH_DTYPE_MAPPING = {
    -1: None,
    0: torch.float32,
    1: torch.float16,
    2: torch.int8,
    3: torch.int32,
    4: torch.uint8,
    6: torch.int16,
    7: None,
    8: None,
    9: torch.int64,
    10: None,
    11: torch.double,
    12: torch.bool
}

logger = get_root_logger()


@BACKEND_WRAPPER.register_module(Backend.CANN.value)
class CANNWrapper(BaseWrapper):

    def __init__(self,
                 model: str,
                 output_names: Optional[Sequence[str]] = None,
                 device: str = 'npu'):
        super().__init__(output_names)
        device = torch.device(device)
        assert device.type == 'npu', \
            f'Expect device type npu, but get {device.type}.'
        self._device_id = device.index if device.index is not None else 0
        with torch_npu.npu.device(self._device_id):
            self.output_names = output_names

            logger.debug(f'Load CANN model from {model}.')
            model_id, ret = acl.mdl.load_from_file(model)
            model_desc = acl.mdl.create_desc()
            ret = acl.mdl.get_desc(model_desc, model_id)
            self._model_id = model_id
            self._model_desc = model_desc

            logger.debug('create input dataset.')
            # get input
            _input_datas = {}
            _input_dataset = acl.mdl.create_dataset()
            input_size = acl.mdl.get_num_inputs(model_desc)
            for i in range(input_size):
                dims_info, ret = acl.mdl.get_input_dims_v2(model_desc, i)
                name = dims_info['name']
                buffer_size = acl.mdl.get_input_size_by_index(model_desc, i)
                buffer = torch.empty([buffer_size // 4],
                                     dtype=torch.float32,
                                     device=torch.device(
                                         'npu', self._device_id))
                data = acl.create_data_buffer(buffer.data_ptr(), buffer_size)
                _, ret = acl.mdl.add_dataset_buffer(_input_dataset, data)
                _input_datas[name] = buffer

            logger.debug('create output dataset.')
            # get output
            _output_datas = {}
            _output_dataset = acl.mdl.create_dataset()
            output_size = acl.mdl.get_num_outputs(model_desc)
            for i in range(output_size):
                dims_info, ret = acl.mdl.get_output_dims(model_desc, i)
                name = dims_info['name']
                buffer_size = acl.mdl.get_output_size_by_index(model_desc, i)
                buffer = torch.empty([buffer_size // 4],
                                     dtype=torch.float32,
                                     device=torch.device(
                                         'npu', self._device_id))
                data = acl.create_data_buffer(buffer.data_ptr(), buffer_size)
                _, ret = acl.mdl.add_dataset_buffer(_output_dataset, data)
                _output_datas[name] = buffer

            self._input_dataset = _input_dataset
            self._output_dataset = _output_dataset
            self._input_datas = _input_datas
            self._output_datas = _output_datas

    def _set_input(self, name: str, val: torch.Tensor):
        """set input value."""
        if val.device.type != 'npu':
            logger.warning(
                f'Expect tensor:{name} device npu, but get {val.device.type}.')
            val = val.to(torch.device('npu', self._device_id))
        assert val.device.index == self._device_id, \
            f'Expect tensor:{name} device id {self._device_id},'\
            f' but get {val.device.index}.'

        buffer = self._input_datas[name]
        buffer_size = buffer.element_size() * buffer.numel()
        val_size = val.element_size() * val.numel()
        assert val_size <= buffer_size, \
            f'Expect tensor:{name} buffer_size <= {buffer_size}, '\
            f'but get {val_size}.'

        stream = torch.npu.current_stream(
            device=torch.device(f'npu:{self._device_id}'))

        val = val.contiguous()
        ret = acl.rt.memcpy_async(buffer.data_ptr(), buffer_size,
                                  val.data_ptr(), val_size,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE,
                                  stream.npu_stream)
        assert ret == 0, 'Input memcpy failed.'

    def _get_output(self, name: str):
        """get output tensor."""
        model_desc = self._model_desc
        index, ret = acl.mdl.get_output_index_by_name(model_desc, name)
        output_dims, ret = acl.mdl.get_cur_output_dims(model_desc, index)
        dims = output_dims['dims']
        acl_dtype = acl.mdl.get_output_data_type(model_desc, index)
        dtype = ACL_TORCH_DTYPE_MAPPING[acl_dtype]
        assert dtype is not None, f'Unknown dtype id {acl_dtype}.'
        output = torch.empty(
            dims, dtype=dtype, device=f'npu:{self._device_id}')
        buffer = self._output_datas[name]
        output_size = output.element_size() * output.numel()

        stream = torch.npu.current_stream(
            device=torch.device(f'npu:{self._device_id}'))
        ret = acl.rt.memcpy_async(output.data_ptr(), output_size,
                                  buffer.data_ptr(), output_size,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE,
                                  stream.npu_stream)
        assert ret == 0, 'Input memcpy failed.'
        return output

    def _set_dynamic_size(self, size: Sequence[int]):
        """Set the input batch size."""
        # get model id and desc
        model_desc = self._model_desc
        model_id = self._model_id

        # check if shape is dynamic
        if 'ascend_mbatch_shape_data' not in self._input_datas:
            return
        index, ret = acl.mdl.get_input_index_by_name(
            model_desc, 'ascend_mbatch_shape_data')

        # check if dynamic batch is supported
        dyna_batch_info, ret = acl.mdl.get_dynamic_batch(model_desc)
        if ret == 0 and dyna_batch_info['batchCount'] != 0:
            batch_size = size[0]

            # if the batch size over the limit.
            assert batch_size <= max(
                dyna_batch_info['batch']), 'Batch size is too large.'

            # set dynamic batch
            ret = acl.mdl.set_dynamic_batch_size(model_id, self._input_dataset,
                                                 index, batch_size)
            assert ret == 0, 'Set batch size failed.'

        # check if dynamic hw is supported
        dyna_hw_info, ret = acl.mdl.get_dynamic_hw(model_desc, -1)
        if ret == 0 and dyna_hw_info['hwCount'] != 0 and len(size) == 4:
            h, w = size[2:]

            # set dynamic hw
            ret = acl.mdl.set_dynamic_hw_size(model_id, self._input_dataset,
                                              index, h, w)
            assert ret == 0, 'Set hw size failed.'

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]]:
        """Run forward inference.

        Args:
            inputs (torch.Tensor | Sequence[torch.Tensor] | Dict[str,
                torch.Tensor]): The input tensor, or tensor sequence, or pairs
                of input names and tensors.

        Return:
            outputs (torch.Tensor | Sequence[torch.Tensor] | Dict[str,
                torch.Tensor]): The input tensor, or tensor sequence, or pairs
                of input names and tensors.
        """
        stream = torch.npu.current_stream(
            device=torch.device(f'npu:{self._device_id}'))

        output = {}
        with torch.npu.stream(stream):
            # set inputs
            for name, val in inputs.items():
                self._set_input(name, val)

            # set batch size
            size = next(iter(inputs.values())).size()
            self._set_dynamic_size(size)

            # set dynamic hw

            # inference
            ret = self.__acl_execute(stream)
            assert ret == 0, 'ACL inference failed.'

            # get outputs
            for name in self._output_datas:
                new_name = name.split(':')[-1]
                output[new_name] = self._get_output(name)

        return output

    @TimeCounter.count_time()
    def __acl_execute(self, stream) -> int:
        """Run inference with CANN.

        Args:
            stream (Stream): the stream to do the inference.
        Returns:
            int: error code, 0 if success.
        """
        return acl.mdl.execute_async(self._model_id, self._input_dataset,
                                     self._output_dataset, stream.npu_stream)
