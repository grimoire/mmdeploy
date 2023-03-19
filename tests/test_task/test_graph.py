# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmdeploy.task.graph import Graph, Node, Pad, Use


class TestUse:

    @pytest.fixture(scope='class')
    def dummy_use(self):
        return Use(Node('from_node'), 'from_id', Node('to_node'), 'to_id')

    def test_operator(self, dummy_use):
        assert dummy_use == Use(dummy_use.from_node, 'from_id',
                                dummy_use.to_node, 'to_id')
        assert dummy_use != Use(dummy_use.from_node, 'from_id',
                                dummy_use.to_node, 'to_id1')

    def test_repr(self, dummy_use):
        assert repr(dummy_use) == 'Use(from_node[from_id] => to_node[to_id])'


class TestPad:

    def test_uses(self):
        node0 = Node('node0')
        pad0 = Pad('pad0', node0)
        node1 = Node('node1')
        pad1 = Pad('pad1', node1)

        assert pad0.unique_id != pad1.unique_id

        pad0.add_use(node1, 'out0')
        pad0.add_use(node1, 'out1')
        assert len(pad0.uses) == 2

        pad0.remove_all_uses()
        assert len(pad0.uses) == 0

        pad1.add_use(node1, 'out0')
        pad1.add_use(node1, 'out1')
        pad0.take_use_from(pad1)
        assert len(pad0.uses) == 2
        assert len(pad1.uses) == 0

    def test_repr(self):
        node0 = Node('node0')
        node1 = Node('node1')
        pad0 = Pad('pad0', node0)
        pad0.add_use(node1, 'out0')
        pad0.add_use(node1, 'out1')

        assert repr(pad0) == ('Pad(name=pad0, owner=node0, '
                              'uses=[Use(node0[pad0] => node1[out0]),'
                              ' Use(node0[pad0] => node1[out1])])')


class TestNode:

    @pytest.fixture(scope='class')
    def from_id(self):
        yield 0

    @pytest.fixture(scope='class')
    def to_id(self):
        yield 1

    @pytest.fixture
    def from_node(self):
        yield Node('from_node')

    @pytest.fixture
    def to_node(self):
        node = Node('to_node')
        yield node

    def test_attribute(self, from_node, to_node, from_id, to_id):
        from_node.add_use(to_node, from_id, to_id)
        assert from_node.name == 'from_node'
        assert from_id in from_node.out_pads
        assert to_id in to_node.in_pads
        assert from_node.use_nodes == [to_node]
        assert to_node.used_nodes == [from_node]

    def test_attrs(self, from_node):
        from_node['node_attr'] = 'attr'
        assert from_node['node_attr'] == 'attr'
        assert from_node.attrs == dict(node_attr='attr')

    def test_pad(self, from_node, to_node, from_id, to_id):
        from_node.add_out_pad(from_id)
        assert from_id in from_node.out_pads
        pad = from_node.out_pads[from_id]
        to_node.add_in_pad(to_id, pad)
        assert to_id in to_node.in_pads
        to_node.remove_in_pad(to_id)
        assert to_id not in to_node.in_pads
        from_node.add_use(to_node, from_id, to_id)
        from_node.remove_out_pad(from_id)
        assert from_id not in from_node.out_pads

    def test_use(self, from_node, to_node, from_id, to_id):
        from_node.add_use(to_node, from_id, to_id)
        assert from_id in from_node.out_pads
        assert to_id in to_node.in_pads
        from_node.remove_all_uses()
        assert from_id in from_node.out_pads
        assert to_id not in to_node.in_pads


class TestGraph:

    @pytest.fixture
    def graph(self):
        yield Graph()

    def test_dump_load(self, graph):
        import pickle
        graph_data = pickle.dumps(graph)
        assert len(graph_data) > 0
        graph = pickle.loads(graph_data)
        assert isinstance(graph, Graph)

    def test_property(self, graph):
        input = graph.add_input('input')
        from_node = graph.add_node('from_node')
        to_node = graph.add_node('to_node')
        output = graph.add_output('output')
        input.add_use(from_node)
        from_node.add_use(to_node)
        to_node.add_use(output)
        assert graph.nodes == dict(
            input=input, from_node=from_node, to_node=to_node, output=output)
        assert graph.inputs == dict(input=input)
        assert graph.outputs == dict(output=output)

    def test_node(self, graph):
        input = graph.add_input('input')
        from_node = graph.add_node('from_node')
        to_node = graph.add_node('to_node')
        output = graph.add_output('output')
        graph.add_use(input, from_node)
        graph.add_use(from_node, to_node)
        graph.add_use(to_node, output)
        assert graph.nodes == dict(
            input=input, from_node=from_node, to_node=to_node, output=output)
        graph.remove_input('input')
        assert graph.inputs == dict()
        graph.remove_output('output')
        assert graph.outputs == dict()
        graph.remove_node('from_node')
        assert graph.nodes == dict(to_node=to_node)

    def test_merge_graph(self):
        graph1 = Graph()
        input1 = graph1.add_input('input1')
        from_node1 = graph1.add_node('from_node1')
        to_node1 = graph1.add_node('to_node1')
        output1 = graph1.add_output('output1')
        graph1.add_use(input1, from_node1)
        graph1.add_use(from_node1, to_node1)
        graph1.add_use(to_node1, output1)

        graph2 = Graph()
        input2 = graph2.add_input('input2')
        from_node2 = graph2.add_node('from_node2')
        to_node2 = graph2.add_node('to_node2')
        output2 = graph2.add_output('output2')
        graph2.add_use(input2, from_node2)
        graph2.add_use(from_node2, to_node2)
        graph2.add_use(to_node2, output2)

        graph = graph1.merge_graph(graph2, {'output1': 'input2'})
        assert 'input1' in graph.inputs
        assert 'input2' not in graph.inputs
        assert 'output1' not in graph.outputs
        assert 'output2' in graph.outputs

    def test_check(self, graph):
        input = graph.add_input('input')
        from_node = graph.add_node('from_node')
        to_node = graph.add_node('to_node')
        output = graph.add_output('output')
        graph.add_use(input, from_node)
        graph.add_use(from_node, to_node)
        graph.add_use(to_node, output)
        assert graph.check()
        from_node.remove_all_uses()
        assert not graph.check()
        assert graph.check(connected=False)
        graph.add_use(from_node, to_node)
        graph.add_use(to_node, from_node, 0, 1)
        assert not graph.check(acyclic=True)
        assert graph.check(acyclic=False)

    def test_topo_sort(self, graph):
        input = graph.add_input('input')
        from_node = graph.add_node('from_node')
        to_node = graph.add_node('to_node')
        output = graph.add_output('output')
        graph.add_use(input, from_node)
        graph.add_use(from_node, to_node)
        graph.add_use(to_node, output)
        sorted_nodes = graph.topo_sort()
        assert len(sorted_nodes) == 4
        assert input in sorted_nodes
        assert from_node in sorted_nodes
        assert to_node in sorted_nodes
        assert output in sorted_nodes
