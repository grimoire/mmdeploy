# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import pytest

from mmdeploy.task.expr.base_expr import BaseExpr, IdentityExpr
from mmdeploy.task.expr.trace import TraceContext, trace_expr
from mmdeploy.task.graph import Graph


class TestTraceContext:

    @pytest.fixture
    def expr(self):

        class MyExpr(BaseExpr):

            def forward(self, x, y=1):
                return x + y

        return MyExpr()

    @pytest.fixture
    def name_expr_map(self, expr):
        return {expr: 'a'}

    @pytest.fixture
    def graph(self):
        return Graph()

    @pytest.fixture
    def context(self, graph, name_expr_map):
        return TraceContext(graph, name_expr_map)

    def test_property(self, context, graph):
        assert context.graph == graph

    def test_scope(self, context, expr):
        with context.scope(expr):
            assert context.cur_scope() == 'a'

        assert context.cur_scope() is None

    def test_add_expr(self, context, graph, expr):
        args = [1]
        ret = expr.forward(*args)
        context.add_inputs(dict(x=args[0]))
        with context.scope(expr):
            context.add_expr(expr, args, {}, ret)

        assert len(graph.nodes) == 2
        assert len(graph.inputs) == 1
        assert graph.check()


@pytest.fixture
def my_expr():

    class MyAdd(BaseExpr):

        def __init__(self):
            super().__init__()

        def forward(self, a, b) -> Any:
            return a + b

    class MyMul(BaseExpr):

        def __init__(self):
            super().__init__()

        def forward(self, a, b) -> Any:
            return a * b

    class MyFMA(BaseExpr):

        def __init__(self):
            super().__init__()
            self.add = MyAdd()
            self.mul = MyMul()

        def atomic(self):
            return False

        def forward(self, a, b, c) -> Any:
            return self.add(self.mul(a, b), c)

    return MyFMA()


def test_trace_expr(my_expr):
    args = [3, 1]
    kwargs = {'c': 2}
    graph = trace_expr(my_expr, args=args, kwargs=kwargs)

    assert graph.check()

    assert len(graph.inputs) == 3
    for _, node in graph.inputs.items():
        assert node['scope'] is None
        assert type(node['expr']) == IdentityExpr

    assert len(graph.outputs) == 1
    for _, node in graph.outputs.items():
        assert node['scope'] is None
        assert type(node['expr']) == IdentityExpr

    assert len(graph.nodes) == 6
