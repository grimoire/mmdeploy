# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from mmdeploy.task.graph import Graph, Node, Pad
from mmdeploy.utils import get_root_logger
from .base_expr import BaseExpr, ConstExpr, IdentityExpr

logger = get_root_logger()

RETURN_ALL = '*'


class TraceContext:
    """Graph trace context.

    Args:
        graph (Graph): The graph object.
        name_expr_map (Dict): Map between expr and name.
        const_builder (Callable): The const expr builder, create when expr
            use it's default argument.
        ident_builder (Callable): The identity expr builder, used to create
            input/output node.
        root_scope (str):
    """

    context_stack = []

    def __init__(self,
                 graph: Graph,
                 name_expr_map: Dict[BaseExpr, str],
                 const_builder: Callable = ConstExpr,
                 ident_builder: Callable = IdentityExpr,
                 root_scope: str = '__root') -> None:
        self._graph: Graph = graph
        self._scopes: list = [None]
        self._expr_outputs: Dict[Any, Pad] = OrderedDict()
        self._name_expr_map = name_expr_map
        self._node_count = 0
        self._const_builder = const_builder
        self._ident_builder = ident_builder
        self._disable_count = 0
        self._root_scope = root_scope

    @property
    def graph(self):
        return self._graph

    @property
    def enable(self):
        return self._disable_count == 0

    @classmethod
    def current_context(cls):
        if len(cls.context_stack) == 0:
            return None
        else:
            return cls.context_stack[-1]

    def __enter__(self) -> 'TraceContext':
        self.context_stack.append(self)
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.context_stack.pop()

    @contextlib.contextmanager
    def disabler(self):
        self._disable_count += 1
        yield
        self._disable_count -= 1

    def cur_scope(self):
        return self._scopes[-1]

    @contextlib.contextmanager
    def scope(self, expr):
        name = self._name_expr_map.get(expr, None)
        self._scopes.append(name)

        yield self._scopes

        self._scopes.pop()

    def add_inputs(self, kwargs):
        graph = self._graph
        expr_outputs = self._expr_outputs
        for k, v in kwargs.items():
            expr = self._ident_builder()
            input_node = graph.add_input(
                k, type='input', scope=None, expr=expr)
            pad = input_node.add_out_pad(RETURN_ALL)
            expr_outputs[id(v)] = pad

    def add_outputs(self, kwargs):
        graph = self._graph
        expr_outputs = self._expr_outputs
        for k, v in kwargs.items():
            assert id(v) in expr_outputs, \
                f'Can not produce output `{k}` from the graph.'
            pad = expr_outputs[id(v)]
            expr = self._ident_builder()
            output_node = graph.add_output(
                k, type='output', scope=None, expr=expr)
            output_node.add_out_pad(RETURN_ALL)
            pad.add_use(output_node, RETURN_ALL)

    def add_consts(self, kwargs):
        graph = self._graph
        expr_outputs = self._expr_outputs
        for k, v in kwargs.items():
            expr = self._const_builder(v)
            input_node = graph.add_input(
                k, type='const', scope=None, expr=expr)
            pad = input_node.add_out_pad(k)
            expr_outputs[id(v)] = pad

    def add_expr(self, expr, args, kwargs: Dict, outputs: Dict):
        graph = self._graph
        expr_outputs = self._expr_outputs
        scope = self._name_expr_map.get(expr, self._root_scope)

        def _add_node(scope, expr):
            node_id = self._node_count
            self._node_count += 1
            node_name = f'{scope}_{node_id}'
            return graph.add_node(
                node_name, type='node', scope=scope, expr=expr)

        def _register_out_pad(node, key, out):
            pad = node.add_out_pad(key)
            expr_outputs[id(out)] = pad

        def _add_use(node, key, arg):
            from_pad = expr_outputs.get(id(arg), None)

            if from_pad is None:
                const_key = scope + '.const'
                self.add_consts({const_key: arg})
                const_node = graph.nodes[const_key]
                _register_out_pad(const_node, RETURN_ALL, arg)
                from_pad = expr_outputs[id(arg)]

            from_pad.add_use(node, key)

        expr_node = _add_node(scope=scope, expr=expr)

        # connect input pad
        for key, arg in enumerate(args):
            _add_use(expr_node, key, arg)

        for key, arg in kwargs.items():
            _add_use(expr_node, key, arg)

        # add output pad
        if isinstance(outputs, (list, tuple)):
            for key, out in enumerate(outputs):
                _register_out_pad(expr_node, key, out)
        elif isinstance(outputs, dict):
            for key, out in outputs.items():
                _register_out_pad(expr_node, key, out)
        _register_out_pad(expr_node, RETURN_ALL, outputs)


def remove_unused_pad(nodes: List[Node]):
    """Remove pads with 0 use.

    Args:
        nodes (List[Node]): Nodes in the graph
    """
    for node in nodes:
        out_pads = node.out_pads
        remove_names = [
            name for name, pad in out_pads.items() if len(pad.uses) == 0
        ]
        for name in remove_names:
            node.remove_out_pad(name)


def trace_expr(expr: BaseExpr,
               args: List = [],
               kwargs: Dict = {},
               input_names: Optional[List[str]] = None,
               output_names: Optional[List[str]] = None,
               root_scope: str = None) -> Graph:
    """Trace the expression, generate a graph.

    Args:
        expr (BaseExpr): The expression to trace
        args (List, optional): The input arguments. Defaults to [].
        kwargs (Dict, optional): The input key value pairs. Defaults to {}.
        input_names (List, optional): The input names. Defaults to None.
        output_names (List, optional): The output names. Defaults to None.
        root_scope (str, optional): The root scope. Defaults to None.

    Returns:
        Graph: Generated graph.
    """

    logger.info(f'Trace expr: {type(expr)}')
    assert len(args) + len(kwargs) > 0, 'Please provide trace inputs.'

    def _get_name_val_pair(args, kwargs, names):
        """generate key value pairs."""
        arg_names = names[:len(args)]
        ret = dict(zip(arg_names, args))
        ret.update(kwargs)

        return ret

    # forward once to get the input/output names.
    expr(*args, **kwargs)

    # set input/output names if not given.
    if input_names is not None:
        assert len(args) + len(kwargs) == len(input_names)
    else:
        input_names = expr.input_names()

    if output_names is None:
        output_names = expr.output_names()

    # initialize graph
    graph = Graph(ret_type=expr.ret_type())

    # Get all sub exprs.
    # We expect root expr is the root ancestor
    name_expr_map = OrderedDict()

    def _register_subexprs(expr: BaseExpr, prefix: str):
        for name, child in expr.named_children():
            submod_qualname = prefix + '.' + name
            name_expr_map[child] = submod_qualname
            _register_subexprs(child, submod_qualname)

    if root_scope is None:
        root_scope = type(expr).__name__
    name_expr_map[expr] = root_scope
    _register_subexprs(expr, name_expr_map[expr])

    # forward in trace context to trace graph.
    with TraceContext(
            graph, name_expr_map, root_scope=root_scope) as trace_context:

        # add graph input
        graph_inputs = _get_name_val_pair(args, kwargs, input_names)
        trace_context.add_inputs(graph_inputs)

        # trace graph
        outputs = expr(*args, **kwargs)

        # add graph output
        if not isinstance(outputs, (list, tuple, dict)):
            outputs = [outputs]

        if isinstance(outputs, dict):
            assert len(output_names) == len(outputs)
            graph_outputs = outputs
        else:
            assert len(output_names) == len(outputs)
            graph_outputs = dict(zip(output_names, outputs))

        trace_context.add_outputs(graph_outputs)

        # clear graph
        remove_unused_pad(graph.compute_nodes.values())

    return graph
