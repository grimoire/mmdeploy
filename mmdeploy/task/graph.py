# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

PadNameType = Union[str, int]


@dataclass
class Use:
    """Describe the io relationship between nodes.

    Args:
        from_node (Node): The upstream node.
        from_id (str|int): The upstream pad id.
        to_node (Node): The downstream node.
        to_id (str|int): The downstream pad id.
    """

    from_node: 'Node'
    from_id: PadNameType
    to_node: 'Node'
    to_id: PadNameType

    def __str__(self) -> str:
        """to string."""
        ret = f'Use({self.from_node.name}[{self.from_id}]'
        ret += ' => '
        ret += f'{self.to_node.name}[{self.to_id}])'
        return ret

    def __repr__(self) -> str:
        """to repr."""
        return str(self)


@dataclass
class Pad:
    """Output of the node."""

    _count_iter = itertools.count()

    name: PadNameType
    owner: 'Node'

    unique_id: int = field(default=0, init=False)
    uses: List[Use] = field(default_factory=list, init=False)

    def __post_init__(self):
        """post init of pad."""
        self.unique_id = next(Pad._count_iter)

    def add_use(self, use_node: 'Node', use_id: PadNameType):
        """Add use to this pad.

        Args:
            use_node (Node): The downstream node of the use
            use_id (PadNameType): The downstream pad name.
        """
        _add_use(self, use_node, use_id)

    def remove_use(self, use_node: 'Node', use_id: PadNameType):
        """Remove use to this pad.

        Args:
            use_node (Node): The downstream node of the use
            use_id (PadNameType): The downstream pad name.
        """
        _remove_use(self, use_node, use_id)

    def remove_all_uses(self):
        """Remove all uses in this pad."""
        for use in self.uses:
            self.remove_use(use.to_node, use.to_id)

    def take_use_from(self, from_pad: 'Pad'):
        """Take uses from other pad.

        Args:
            from_pad (Pad): All uses in this pad will be taken
        """
        uses = from_pad.uses
        for use in uses:
            to_node = use.to_node
            to_id = use.to_id
            from_pad.remove_use(to_node, to_id)
            self.add_use(to_node, to_id)

    def _add_use_unsafe(self, use: 'Node', use_id: PadNameType):
        """Add use to this pad.

        Do not call it manually
        """
        self.uses.append(Use(self.owner, self.name, use, use_id))

    def _remove_use_unsafe(self, use: 'Node', use_id: PadNameType):
        """Remove use to this pad.

        Do not call it manually
        """
        find_use = Use(self.owner, self.name, use, use_id)
        self.uses = list(filter(lambda u: u != find_use, self.uses))

    def __str__(self) -> str:
        """to string."""
        ret = f'Pad(name={self.name}, owner={self.owner.name}'
        if len(self.uses) > 0:
            ret += ', uses=['
            uses_str = []
            for use in self.uses:
                uses_str += [f'{use}']
            ret += ', '.join(uses_str)
            ret += ']'
        ret += ')'
        return ret

    def __repr__(self) -> str:
        """to repr."""
        return str(self)


class Node:
    """Node object in the graph.

    Args:
        name (str): The name of the node.
    """

    def __init__(self, name: str, **kwargs) -> None:
        self._name: str = name
        self._attrs: Dict = kwargs
        self._in_pads: Dict[Any, Pad] = OrderedDict()
        self._out_pads: Dict[Any, Pad] = OrderedDict()

    @property
    def name(self) -> str:
        """return node name."""
        return self._name

    @property
    def in_pads(self) -> Dict[Any, Pad]:
        """input pads."""
        return self._in_pads

    @property
    def out_pads(self) -> Dict[Any, Pad]:
        """output pads."""
        return self._out_pads

    @property
    def uses(self) -> List[Use]:
        """The uses of the output pads."""
        ret = []
        for _, pad in self.out_pads.items():
            pad_uses = pad.uses
            ret += pad_uses

        return ret

    @property
    def use_nodes(self) -> List['Node']:
        """The nodes use the output of self."""
        uses = self.uses
        return list(set(u.to_node for u in uses))

    @property
    def useds(self) -> List[Use]:
        """The uses of the input pads."""
        ret = []
        in_pads = self.in_pads
        for _, pad in in_pads.items():
            uses = pad.uses
            for use in uses:
                if use.to_node == self:
                    ret.append(use)

        return ret

    @property
    def used_nodes(self) -> List['Node']:
        """The nodes provide the inputs of self."""
        useds = self.useds
        return list(set(u.from_node for u in useds))

    @property
    def attrs(self) -> Dict:
        """The attributes bound to this node."""
        return self._attrs

    def add_in_pad(self, name: PadNameType, pad: Pad):
        """Add input pad to this node.

        Args:
            name (PadNameType): The input pad name.
            pad (Pad): The input pad, comes from another node.
        """
        _add_use(pad, self, name)

    def add_out_pad(self, name: PadNameType, pad: Optional[Pad] = None):
        """Add output pad to this node.

        Args:
            name (PadNameType): The output pad name.
            pad (Optional[Pad], optional): The output pad, if not given,
                a default node would be created.
        """
        assert name not in self.out_pads, (f'An output pad with name: {name}'
                                           ' already exists in the node.')
        if pad is None:
            pad = Pad(name, self)
        self.out_pads[name] = pad
        return pad

    def remove_in_pad(self, name: PadNameType):
        """Remove input pad by name.

        Args:
            name (PadNameType): The input pad name.
        """
        assert name in self.in_pads, f'Input pad: {name} not found.'
        pad = self.in_pads[name]
        _remove_use(pad, self, name)

    def remove_out_pad(self, name: PadNameType):
        """Remove output pad by name.

        Args:
            name (PadNameType): The output pad name.
        """
        assert name in self.out_pads, f'Output pad: {name} not found.'
        pad = self.out_pads.pop(name)
        pad.remove_all_uses()

    def add_use(self,
                use_node: 'Node',
                output_name: PadNameType = 0,
                use_input_name: PadNameType = 0):
        """Connect a downstream node.

        Args:
            use_node (Node): The downsteam node.
            output_name (PadNameType, optional): The output pad name.
                Defaults to 0.
            use_input_name (PadNameType, optional): The input pad name of the
                downsteam node. Defaults to 0.
        """
        if output_name not in self.out_pads:
            self.add_out_pad(output_name)
        pad = self.out_pads[output_name]
        pad.add_use(use_node, use_input_name)

    def remove_all_uses(self):
        """Remove all use of the output."""
        for _, pad in self.out_pads.items():
            pad.remove_all_uses()

    def set_attr(self, name: str, attr: Any):
        """set attribute by name."""
        self.attrs[name] = attr

    def __setitem__(self, name: str, attr: Any):
        """set attribute by name."""
        self.set_attr(name, attr)

    def __getitem__(self, name: str):
        """get attribute by name."""
        return self.attrs[name]

    def __str__(self) -> str:
        """to string."""
        in_pad_names = [
            f'${pad.owner.name}[{pad.name}]' for pad in self.in_pads.values()
        ]
        out_pad_names = [f'${self.name}[{name}]' for name in self.out_pads]

        out_pad_str = ', '.join(out_pad_names)
        in_pad_str = ', '.join(in_pad_names)
        return f'{out_pad_str} = {self.name}({in_pad_str})'

    def __repr__(self) -> str:
        """to repr."""
        return str(self)


class GraphInput(Node):
    """Graph input Node."""
    pass


class GraphOutput(Node):
    """Graph output Node."""
    pass


class Graph:
    """The Graph definition."""

    def __init__(self, **kwargs) -> None:
        self._nodes: Dict[Any, Node] = OrderedDict()
        self._inputs = OrderedDict()
        self._outputs = OrderedDict()
        self._attrs: Dict = kwargs

    @property
    def nodes(self) -> Dict[Any, Node]:
        """Get all nodes in the graph, include input and output."""
        return self._nodes

    @property
    def attrs(self) -> Dict:
        """Get attributes of the graph."""
        return self._attrs

    def set_attr(self, name: str, attr: Any):
        """Set attribute of the graph."""
        self.attrs[name] = attr

    def __setitem__(self, name: str, attr: Any):
        """Set attribute of the graph."""
        self.set_attr(name, attr)

    def __getitem__(self, name: str):
        """Get attribute of the graph."""
        return self.attrs[name]

    @property
    def compute_nodes(self):
        """Get all compute nodes(w/o Input/Output)"""
        input_names = list(self.inputs.keys())
        output_names = list(self.outputs.keys())
        exclude_names = input_names + output_names
        return OrderedDict(
            (k, v) for k, v in self.nodes.items() if k not in exclude_names)

    @property
    def inputs(self) -> Dict[str, GraphInput]:
        """Input nodes."""
        return self._inputs

    @property
    def outputs(self) -> Dict[str, GraphOutput]:
        """Output nodes."""
        return self._outputs

    def add_input(self, name: str, **attrs) -> GraphInput:
        """Add input to the graph by name.

        Args:
            name (str): The input node name.

        Returns:
            GraphInput: The added input node.
        """
        assert name not in self.inputs, (f'Input node: {name} already exist'
                                         ' in the graph.')
        node = self.add_node(name, GraphInput(name), **attrs)
        self.inputs[name] = node
        return node

    def remove_input(self, name: str) -> GraphInput:
        """Remove input to the graph by name.

        Args:
            name (str): The input node name.

        Returns:
            GraphInput: The removed input node.
        """
        assert name in self.inputs, (f'Can not find input node: {name}')
        self.remove_node(name)
        return self.inputs.pop(name)

    def add_output(self, name: str, **attrs) -> GraphOutput:
        """Add output to the graph by name.

        Args:
            name (str): The output node name.

        Returns:
            GraphOutput: The added output node.
        """
        node = self.add_node(name, GraphOutput(name), **attrs)
        return self.mark_output(name, node)

    def mark_output(self, name: str, node: GraphOutput) -> GraphOutput:
        """Mark the output of the graph.

        Args:
            name (str): The name of the output node
            node (GraphOutput): The output node

        Returns:
            GraphOutput: The output node
        """
        assert name not in self.outputs, (f'Output node: {name} already exist'
                                          ' in the graph.')
        self.outputs[name] = node
        return node

    def remove_output(self, name: str) -> GraphOutput:
        """Remove output to the graph by name.

        Args:
            name (str): The output node name.

        Returns:
            GraphOutput: The removed output node.
        """
        assert name in self.outputs, (f'Can not find output node: {name}')
        self.remove_node(name)
        return self.outputs.pop(name)

    def add_node(self,
                 name: str,
                 node: Optional[Node] = None,
                 **attrs) -> Node:
        """Add new node to the graph.

        Args:
            name (str): The node name.
            node (Node, optional): The node object. If not given, a
                default Node object would be generated.

        Returns:
            Node: The added node.
        """
        assert name not in self.nodes, (
            f'Node: {name} already exist in the graph.')
        if node is not None:
            assert node.name == name, (
                f'Node name is {node.name} instead of {name}')
            for k, v in attrs.items():
                node.set_attr(k, v)
        else:
            node = Node(name, **attrs)
        self.nodes[name] = node
        return node

    def remove_node(self, name: str) -> Node:
        """Remove new node to the graph.

        Args:
            name (str): The node name.

        Returns:
            Node: The removed node.
        """
        assert name in self.nodes, f'Can not find node: {name}.'
        return self.nodes.pop(name)

    def add_nodes_from(self, nbunch: Dict):
        """Add nodes.

        Args:
            nbunch (Dict): named nodes.
        """
        for name, node in nbunch.items():
            self.add_node(name, node)

    def add_use(self,
                owner: Node,
                use: Node,
                owner_output_name: PadNameType = 0,
                use_input_name: PadNameType = 0):
        """Add use between two node.

        Args:
            owner (Node): The upstream node.
            user (Node): The downsteam node.
            owner_output_name (PadNameType, optional): The owner output pad
                name.
            user_input_name (PadNameType, optional): The user input pad name.
        """
        owner.add_use(use, owner_output_name, use_input_name)

    def merge_graph(self, other_graph, io_mapping: Dict) -> 'Graph':
        return merge_graph(self, other_graph, io_mapping)

    def _dfs(self,
             node: Node,
             visited: Dict,
             callback: Callable = lambda _: True):
        """DFS traverse implementation.

        Args:
            node (Node): current visit node.
            visited (Dict): The visited map.
            callback (Callable): callback after all sub nodes visited.
        """
        # 0: not visited
        # 1: ancestor
        # 2: visited
        visited[node.name] = 1
        use_nodes = node.use_nodes
        for use in use_nodes:
            if visited[use.name] == 0:
                self._dfs(use, visited, callback)
            elif visited[use.name] == 1:
                # ancestor has been visited
                raise GraphAcyclicError(f'{use.name} has been visited again.')
        visited[node.name] = 2
        callback(node)

    def dfs(self, callback: Callable = lambda _: True):
        """DFS traverse.

        Args:
            callback (Callable): callback after all sub nodes visited.
        """
        visited = OrderedDict((name, 0) for name in self.nodes)
        for _, inp in self.inputs.items():
            self._dfs(inp, visited=visited, callback=callback)

    def check(self, connected: bool = True, acyclic: bool = True) -> bool:
        """Check if the graph is valid.

        Args:
            connected (bool, optional): Check if graph is connected.
            acyclic (bool, optional): Check if the graph contain cycle.

        Returns:
            bool: check result.
        """

        try:
            self.topo_sort()
        except GraphAcyclicError:
            if acyclic:
                return False
        except GraphConnectedError:
            if connected:
                return False

        return True

    def topo_sort(self) -> List[Node]:
        """Perform topo sort on the graph.

        Raises:
            GraphConnectedError: Graph is not connected.

        Returns:
            List[Node]: The sorted nodes.
        """
        ret = []

        def _callback(node):
            nonlocal ret
            ret.append(node)

        self.dfs(_callback)

        if len(ret) < len(self.nodes):
            raise GraphConnectedError(
                'Some nodes can not been visited from inputs.')

        ret.reverse()

        return ret

    def __str__(self) -> str:
        """to string."""
        input_names = list(f'${name}' for name in self.inputs.keys())
        output_names = list(f'${name}' for name in self.outputs.keys())
        nodes = self.compute_nodes

        input_str = ', '.join(input_names)
        output_str = ','.join(output_names)

        head = f'graph({input_str}) -> {output_str}:\n'
        body = [f'    {node}' for node in nodes.values()]
        outputs = [f'    {node}' for node in self.outputs.values()]

        return head + '\n'.join(body) + '\n' + '\n'.join(outputs)

    def __repr__(self) -> str:
        """to repr."""
        return str(self)


class GraphAcyclicError(Exception):
    """The graph acyclic error."""


class GraphConnectedError(Exception):
    """The graph connection error."""


def merge_graph(graph1: Graph, graph2: Graph, io_mapping: Dict) -> Graph:
    """Merge two graph.

        outputs of graph1 would be map to inputs of graph2

    Args:
        graph1 (Graph): The base graph.
        graph2 (Graph): The other graph.
        io_mapping (Dict): The mapping between output of graph1 and input of
            graph2.

    Returns:
        Graph: The merged graph.
    """

    # add all compute nodes in graph2 into graph1
    count = 0
    for node_name, node in graph2.compute_nodes.items():
        new_name = node_name
        while new_name in graph1.nodes:
            new_name = f'{node_name}_{count}'
            count += 1

        graph1.add_node(new_name, node)

    # connect io
    for output_name, input_name in io_mapping.items():
        assert output_name in graph1.outputs
        assert input_name in graph2.inputs

        # get output if graph1 and input of graph2
        output_node: Node = graph1.outputs[output_name]
        input_node: Node = graph2.inputs[input_name]

        # get the output income pad `g1_remain<g1_pad> -> out`
        # and the input outcome pad `in<g2_pad> -> g2_remain`
        g1_pad = next(iter(output_node.in_pads.values()))
        g2_pad = next(iter(input_node.out_pads.values()))

        # disconnect `g1_remain<g1_pad>  out`
        g1_pad.remove_use(output_node, output_name)

        # replace pad `g1_remain<g1_pad> -> g2_remain`
        g1_pad.take_use_from(g2_pad)

        # remove input/output
        graph1.remove_output(output_name)
        graph2.remove_input(input_name)

    # add output of graph2 to graph1
    output_names = list(graph2.outputs.keys())
    for output_name in output_names:
        output_node = graph2.outputs[output_name]
        graph1.mark_output(output_name, output_node)
        graph2.remove_output(output_name)

    return graph1


def _add_use(pad: Pad, use: Node, use_id: PadNameType):
    """Implementation of add use."""
    pad._add_use_unsafe(use, use_id)
    assert use.in_pads.get(use_id, pad) == pad
    if use_id not in use.in_pads:
        use.in_pads[use_id] = pad


def _remove_use(pad: Pad, use: Node, use_id: PadNameType):
    """Implementation of remove use."""
    pad._remove_use_unsafe(use, use_id)
    if use_id in use.in_pads:
        return use.in_pads.pop(use_id)
