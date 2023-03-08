# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
from abc import ABCMeta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BaseIRParam:
    """Base ir param.

    Args:
        args (Any): The arguments of the model.
        work_dir (str): The working directory to save the output.
        file_name (str): The file name of the output. postfix can be omitted.
        input_names (List[str]): The names to assign to the input of the ir.
        output_names (List[str]): The names to assign to the output of the ir.
        dynamic_axes (Dict): Determine the dynamic axes of the inputs. It not
            given, all axes will be static.
        backend (str): The expected backend of the ir.
        rewrite_context (Dict): Provide information to the rewriter.
    """
    # latent fields
    _default_postfix = ''
    _file_name = None
    _manager = None

    # class fields
    args: Any = None
    work_dir: str = None
    file_name: str = None
    input_names: List[str] = None
    output_names: List[str] = None
    dynamic_axes: Dict = None
    backend: str = 'default'
    rewrite_context: Dict = field(default_factory=dict)

    @property
    def file_name(self) -> str:
        """file_name getter."""
        return self._file_name

    @file_name.setter
    def file_name(self, val) -> None:
        """file_name setter."""
        if osp.splitext(val)[1] == '':
            val = val + self._default_postfix

        self._file_name = val

    @classmethod
    def get_manager(cls):
        """manager of the ir."""
        return cls._manager

    def check(self):
        """check if the param is valid."""


class BaseIRManager(metaclass=ABCMeta):
    """Abstract interface of ir manager."""

    build_param = BaseIRParam

    @classmethod
    def export(cls, model: Any, *args, **kwargs):
        """export model to ir."""
        raise NotImplementedError(
            'class method: `export` of '
            f'{cls.__qualname__} has not been implemented.')

    @classmethod
    def export_from_param(cls, model, param: BaseIRParam):
        """export model to ir by param."""
        raise NotImplementedError(
            'class method: `export_from_param` of '
            f'{cls.__qualname__} has not been implemented.')

    @classmethod
    def is_available(cls) -> bool:
        """check if the export tools is available."""
        raise NotImplementedError(
            'class method: `is_available` of '
            f'{cls.__qualname__} has not been implemented.')


class IRManagerRegistry:
    """ir manager registry."""

    def __init__(self):
        self._module_dict = {}

    def register(self,
                 name: str,
                 enum_name: Optional[str] = None,
                 param: Any = None):
        """register ir manager.

        Args:
            name (str): name of the ir
            enum_name (Optional[str], optional): enum name of the ir.
                if not given, the upper case of name would be used.
        """
        from mmdeploy.utils import get_root_logger
        logger = get_root_logger()

        if enum_name is None:
            enum_name = name.upper()

        def wrap_manager(cls):

            from mmdeploy.utils import IR

            if not hasattr(IR, enum_name):
                from aenum import extend_enum
                extend_enum(IR, enum_name, name)
                logger.info(f'Registry new ir: {enum_name} = {name}.')

            if name in self._module_dict:
                logger.info(
                    f'Backend manager of `{name}` has already been registered.'
                )

            self._module_dict[name] = cls

            cls.ir_name = name
            cls.build_param = param
            if param is not None:
                param._manager = cls

            return cls

        return wrap_manager

    def find(self, name: str) -> BaseIRManager:
        """Find the ir manager with name.

        Args:
            name (str): ir name.
        Returns:
            BaseBackendManager: ir manager of the given ir.
        """
        # try import name if it exists in `mmdeploy.ir`
        try:
            importlib.import_module(f'mmdeploy.ir.{name}')
        except Exception:
            from mmdeploy.utils import get_root_logger
            logger = get_root_logger()
            logger.debug(f'can not find IR: {name} in `mmdeploy.ir`')
        return self._module_dict.get(name, None)


IR_MANAGERS = IRManagerRegistry()


def get_ir_manager(name: str) -> BaseIRManager:
    """Get ir manager.

    Args:
        name (str): name of the ir.
    Returns:
        BaseIRManager: The ir manager of given name
    """
    from enum import Enum
    if isinstance(name, Enum):
        name = name.value
    return IR_MANAGERS.find(name)