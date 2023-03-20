# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from tempfile import TemporaryDirectory

import pytest

from mmdeploy.task.expr.base_expr import (BaseBlock, BaseExpr, ConstExpr,
                                          ExprContext, IdentityExpr, dump,
                                          load)
from mmdeploy.task.expr.constant import ExprMode


class TestExprContext:

    def test_context(self):
        work_dir_name = 'my_work_dir'
        with ExprContext.from_work_dir(work_dir_name) as env:
            assert env.work_dir == work_dir_name
            assert ExprContext.work_dir() == work_dir_name

        assert ExprContext.work_dir() == '.'


class DummySeqExpr(BaseExpr):

    def forward(self, x, y):
        return x + y, x - y


class DummyDictExpr(BaseExpr):

    def __init__(self):
        super().__init__()
        self._meta_val = 0
        self._lazy_val = 0

    def lazy_init(self):
        self._lazy_val = 1

    def load(self, state, work_dir: str = ...):
        state['_meta_val'] += 1
        return state

    def dump(self, state, work_dir: str = ...):
        state['_meta_val'] += 1
        return state

    def forward(self, x, y):
        return {'out0': x + y, 'out1': x - y}


class TestExpr:

    @pytest.fixture(scope='class')
    def seq_expr(self):
        return DummySeqExpr()

    @pytest.fixture(scope='class')
    def dict_expr(self):
        return DummyDictExpr()

    def test_lazy_init(self):
        expr = DummyDictExpr()
        assert expr._lazy_val == 0

        expr(1, 2)
        assert expr._lazy_val == 1

    def test_io_name(self, seq_expr, dict_expr):
        args = [1, 2]
        seq_expr(*args)

        assert seq_expr.input_names() == ['x', 'y']
        assert seq_expr.output_names() == [0, 1]

        ret = dict_expr(*args)

        assert dict_expr.input_names() == ['x', 'y']
        assert dict_expr.output_names() == ['out0', 'out1']

        # test _generate_io_named_map
        i_map, o_map = dict_expr._generate_io_named_map(args, {}, ret)
        assert i_map == dict(x=1, y=2)
        assert o_map == dict(out0=3, out1=-1)

    def test_save_load(self, dict_expr):
        with TemporaryDirectory() as work_dir:
            file_name = 'tmp.pth'
            assert dict_expr._meta_val == 0
            dump(dict_expr, work_dir, file_name)
            assert osp.exists(osp.join(work_dir, file_name))

            new_expr = load(work_dir, file_name)
            assert isinstance(new_expr, BaseExpr)
            assert new_expr._meta_val == 2

    def test_hook(self, dict_expr):

        def pre_hook(mod, args, kwargs):
            return [val + 1 for val in args], {}

        def post_hook(mod, args, kwargs, output):
            return dict((k, v + 1) for k, v in output.items())

        pre_handle = dict_expr.register_pre_forward_hook(pre_hook)
        post_handle = dict_expr.register_forward_hook(post_hook)

        ret = dict_expr(1, 2)

        assert ret == dict(out0=6, out1=0)

        pre_handle.remove()
        post_handle.remove()

        assert len(dict_expr._pre_forward_hooks) == 0
        assert len(dict_expr._forward_hooks) == 0


def test_const_expr():
    expr = ConstExpr(3)

    assert expr() == 3


def test_identity_expr():
    expr = IdentityExpr()

    assert expr(x=5) == 5
    assert expr(3) == 3


class DummyBlock(BaseBlock):

    def __init__(self):
        super().__init__()

        self.seq_expr = DummySeqExpr()
        self.dict_expr = DummyDictExpr()

    def forward(self, x, y):
        x, y = self.seq_expr(x, y)
        return self.dict_expr(x, y)


class TestBlock:

    @pytest.fixture(scope='class')
    def dummy_block(self):
        return DummyBlock()

    def test_mode(self, dummy_block):
        dummy_block.mode(ExprMode.EXPORT)
        assert dummy_block.mode() == ExprMode.EXPORT
        assert dummy_block.seq_expr.mode() == ExprMode.EXPORT
        assert dummy_block.dict_expr.mode() == ExprMode.EXPORT

        dummy_block.mode(ExprMode.INFERENCE)
        assert dummy_block.mode() == ExprMode.INFERENCE
        assert dummy_block.seq_expr.mode() == ExprMode.INFERENCE
        assert dummy_block.dict_expr.mode() == ExprMode.INFERENCE
