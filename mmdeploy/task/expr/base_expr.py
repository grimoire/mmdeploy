# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import os
import os.path as osp
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from mmdeploy.utils import get_root_logger
from .constant import ExprMode, ReturnType
from .utils import RemovableHandle

logger = get_root_logger()


@dataclass
class ExprEnv:
    """The expression environment.

    Args:
        work_dir (str): The working directory.
    """
    work_dir: str = '.'


class ExprContext:
    """The expression environment context.

    Args:
        ExprEnv (str): New environment
    """
    _env_stack: List[ExprEnv] = [ExprEnv()]

    def __init__(self, env: ExprEnv) -> None:
        self._cur_env: ExprEnv = env

    @classmethod
    def from_work_dir(cls, work_dir: str) -> 'ExprContext':
        """build context from work directory.

        Args:
            work_dir (str): The working directory.

        Returns:
            ExprContext: The expression context
        """
        return ExprContext(ExprEnv(work_dir=work_dir))

    @classmethod
    def env(cls) -> ExprEnv:
        """Get current expression environment."""
        assert len(cls._env_stack) > 0, ''
        return cls._env_stack[-1]

    @classmethod
    def work_dir(cls) -> str:
        """Get current work directory."""
        return cls.env().work_dir

    def __enter__(self) -> ExprEnv:
        """Enter the context."""
        self._env_stack.append(self._cur_env)
        return self.env()

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        """Leave the context."""
        self._env_stack.pop()


def load(work_dir: str, expr_file: str = '_exp.pth') -> Any:
    """load object from work directory.

    Args:
        work_dir (str): The work directory.
        expr_file (str, optional): The save file of. Defaults to '_exp.pth'.

    Returns:
        Any: _description_
    """
    load_path = osp.join(work_dir, expr_file)
    logger.info(f'Load expr from {load_path}')
    if not osp.exists(work_dir):
        os.mkdir(work_dir)
    with ExprContext.from_work_dir(work_dir):
        with open(load_path, 'rb') as f:
            return pickle.load(f)


def dump(obj: Any, work_dir: str, expr_file: str = '_exp.pth'):
    save_path = osp.join(work_dir, expr_file)
    logger.info(f'Save expr to {save_path}')
    if not osp.exists(work_dir):
        os.mkdir(work_dir)
    with ExprContext.from_work_dir(work_dir):
        with open(save_path, 'wb') as f:
            pickle.dump(obj, f)


class BaseExpr:
    """The Expression base class."""

    def __init__(self):
        self._named_children: List[Tuple[str, BaseExpr]] = []
        self._pre_forward_hooks: Dict[int, RemovableHandle] = OrderedDict()
        self._forward_hooks: Dict[int, RemovableHandle] = OrderedDict()
        self._traced_input_names: Optional[List] = None
        self._traced_output_names: Optional[List] = None
        self._mode: ExprMode = ExprMode.EXPORT
        self._finish_lazy_init: bool = False
        self._ret_type: ReturnType = ReturnType.UNKNOWN

    def input_names(self):
        """The input names of the expression."""
        assert self._traced_input_names is not None, (
            f'Can not get input names of {type(self)}. '
            'Call this expr to trace the names.')
        return self._traced_input_names

    def output_names(self):
        """The output names of the expression."""
        assert self._traced_output_names is not None, (
            f'Can not get output names of {type(self)}. '
            'Call this expr to trace the names.')
        return self._traced_output_names

    def ret_type(self):
        """The return type of the expression."""
        return self._ret_type

    def register_pre_forward_hook(self, hook: Callable) -> RemovableHandle:
        """Register pre-forward hook.

        Args:
            hook (Callable): The pre-forward hook

        Returns:
            RemovableHandle: The handle of the hook
        """
        handle = RemovableHandle(self._pre_forward_hooks)
        self._pre_forward_hooks[handle.id] = hook
        return handle

    def register_forward_hook(self, hook: Callable) -> RemovableHandle:
        """Register forward hook.

        Args:
            hook (Callable): The forward hook

        Returns:
            RemovableHandle: The handle of the hook
        """
        handle = RemovableHandle(self._forward_hooks)
        self._forward_hooks[handle.id] = hook
        return handle

    def atomic(self):
        """Return true if the expression does not contain subgraph."""
        return True

    def _propagate_to_children(self,
                               callback: Callable,
                               non_atomic_only: bool = False):
        """Propagate the callback to all children in the expression.

        Args:
            callback (Callable): The callback function.
            non_atomic_only (bool, optional): Only propagate if
                self.atomic == False.
        """
        if not (non_atomic_only and self.atomic()):
            for _, child in self.named_children():
                callback(child)

    def mode(self, new_mode: ExprMode = None) -> ExprMode:
        """Get or set the expression mode.

        Args:
            new_mode (ExprMode, optional): The update expression mode.
                If not given, return current mode.

        Returns:
            ExprMode: updated mode
        """
        self._propagate_to_children(lambda child: child.mode(new_mode))
        if new_mode is not None:
            self._mode = new_mode
        return self._mode

    def named_children(self) -> List[Tuple[str, 'BaseExpr']]:
        """Return all children in this expression."""
        return self._named_children

    def __setattr__(self, __name: str, __value: Any) -> None:
        """Set attribute."""
        if isinstance(__value, BaseExpr):
            if not hasattr(self, '_named_children'):
                raise AttributeError(
                    f'Can not add expr in {type(self).__qualname__}. '
                    'have you init the expr with super().__init__()?')
            self._named_children.append((__name, __value))
        super().__setattr__(__name, __value)

    def __getstate__(self):
        """get state."""
        state = self.__dict__.copy()
        del state['_forward_hooks']
        del state['_pre_forward_hooks']
        state = self.dump(state, ExprContext.work_dir())
        return state

    def __setstate__(self, state):
        """set state."""
        state = self.load(state, ExprContext.work_dir())
        self.__dict__ = state
        state['_finish_lazy_init'] = False
        self._pre_forward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()

    def load(self, state: Dict, work_dir: str = ExprContext.work_dir()):
        """Callback in setstate.

        Args:
            state (Dict): state of the object.
            work_dir (str): the working directory.
        """
        return state

    def dump(self, state: Dict, work_dir: str = ExprContext.work_dir()):
        """Callback in getstate.

        Args:
            state (Dict): state of the object.
            work_dir (str): the working directory.
        """
        return state

    def dump_info(self, work_dir: str, meta_info: Dict):
        """Dump meta info.

        Args:
            work_dir (str): the working directory.
            meta_info (Dict): the meta information. will be save as json files.
        """
        self._propagate_to_children(
            lambda child: child.dump_info(work_dir, meta_info))

    def lazy_init(self):
        """lazy init."""
        pass

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """forward of the expression."""
        raise NotImplementedError(
            f'forward has not been implemented for expr {type(self)}.')

    # def _slow_forward(self, *args: Any, **kwargs: Any) -> Any:
    #     from .trace import TraceContext
    #     trace_context = TraceContext.current_context()
    #     if trace_context is not None:
    #         with trace_context.scope(self):
    #             if self.atomic() and trace_context.enable:
    #                 with trace_context.disabler():
    #                     outputs = self.forward(*args, **kwargs)
    #                 kw_inputs, _ = self._generate_io_named_map(
    #                     args, kwargs, outputs)
    #                 trace_context.add_expr(self, [], kw_inputs, outputs)
    #             else:
    #                 outputs = self.forward(*args, **kwargs)
    #                 self._parse_io_names(args, kwargs, outputs)
    #     else:
    #         outputs = self.forward(*args, **kwargs)
    #         self._parse_io_names(args, kwargs, outputs)
    #     return outputs

    def _call_impl(self, *args, **kwargs):
        """call implementation."""

        # lazy init
        if not self._finish_lazy_init:
            self.lazy_init()
            self._finish_lazy_init = True

        # pre forward hook
        for hook in self._pre_forward_hooks.values():
            hook_results = hook(self, args=args, kwargs=kwargs)
            if hook_results is not None:
                args, kwargs = hook_results

        # from .trace import TraceContext
        # trace_context = TraceContext.current_context()
        # forward_call = self._slow_forward if trace_context is not None \
        #     else self.forward
        forward_call = self.forward
        results = forward_call(*args, **kwargs)
        self._parse_io_names(args, kwargs, results)

        # forward hook
        for hook in self._forward_hooks.values():
            hook_results = hook(self, args=args, kwargs=kwargs, output=results)
            if hook_results is not None:
                results = hook_results
        return results

    __call__ = _call_impl

    def _parse_input_names(self,
                           args: Sequence,
                           kwargs: Dict,
                           forward_func: Callable = None):
        """parse input names from function.

        Args:
            args (Sequence): input arguments
            kwargs (Dict): input key value pairs
            forward_func (Callable): Callable function to parse
        """
        if forward_func is None:
            forward_func = self.forward

        # generate input names
        signature = inspect.signature(forward_func)
        parameters = signature.parameters
        bound = signature.bind(*args, **kwargs)
        arguments = bound.arguments

        self._traced_input_names = []
        for name, val in arguments.items():
            param = parameters[name]
            if param.kind == inspect._ParameterKind.VAR_POSITIONAL:
                for idx in range(len(val)):
                    self._traced_input_names.append(f'{name}__{idx}')
            elif param.kind == inspect._ParameterKind.VAR_KEYWORD:
                for kwname in val:
                    self._traced_input_names.append(kwname)
            else:
                self._traced_input_names.append(name)

    def _parse_output_names(self, outputs: Any):
        """parse output names from outputs.

        Args:
            outputs (Any): function outputs
        """
        if isinstance(outputs, (tuple, list)):
            self._ret_type = ReturnType.SEQUENCE
            self._traced_output_names = []
            for i in range(len(outputs)):
                self._traced_output_names.append(i)
        elif isinstance(outputs, dict):
            self._ret_type = ReturnType.DICT
            self._traced_output_names = list(outputs.keys())
        else:
            self._ret_type = ReturnType.OTHER
            self._traced_output_names = [0]

    def _parse_io_names(self,
                        args: Sequence,
                        kwargs: Dict,
                        outputs: Any,
                        forward_func: Callable = None):
        """parse input and output names.

        Args:
            args (Sequence): input arguments
            kwargs (Dict): input key value pairs
            outputs (Any): function outputs
            forward_func (Callable): Callable function to parse
        """
        if forward_func is None:
            forward_func = self.forward

        # generate input names
        self._parse_input_names(args, kwargs, forward_func=forward_func)

        # generate output names
        self._parse_output_names(outputs)

    def _generate_io_named_map(self, args: Sequence, kwargs: Dict,
                               outputs: Any) -> Tuple[Dict, Dict]:
        """convert input and output to key value pairs.

        Args:
            args (Sequence): positional arguments.
            kwargs (Dict): keyword arguments.
            outputs (Any): outputs.

        Returns:
            Tuple[Dict, Dict]: inputs and outputs mapping.
        """

        # generate io names
        self._parse_io_names(args, kwargs, outputs)

        # generate input kv
        kw_inputs = self._bind_sig(*args, **kwargs)

        # generate output kv
        if isinstance(outputs, dict):
            kw_outputs = outputs
        elif isinstance(outputs, (list, tuple)):
            kw_outputs = dict(zip(self.output_names(), outputs))
        else:
            kw_outputs = dict(zip(self.output_names(), [outputs]))
        return kw_inputs, kw_outputs

    def _bind_sig(self, *args, **kwargs):
        """bind signature."""
        import inspect
        from inspect import Parameter, Signature
        kind = inspect._ParameterKind.POSITIONAL_OR_KEYWORD
        input_names = self.input_names()
        sig = Signature([Parameter(name, kind) for name in input_names])
        bind = sig.bind(*args, **kwargs)
        return bind.arguments


class ConstExpr(BaseExpr):
    """Const expression."""

    def __init__(self, val: Any) -> None:
        super().__init__()
        self._val = val

    def forward(self):
        return self._val


class IdentityExpr(BaseExpr):
    """Identity expression."""

    def forward(self, *args, **kwargs):
        assert (len(args) + len(kwargs)) == 1
        if len(args) > 0:
            return args[0]
        else:
            for v in kwargs.values():
                return v


class BaseBlock(BaseExpr):
    """Block."""

    def atomic(self):
        return False
