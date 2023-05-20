
from typing import List, Optional

from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode


def dag_layers(dag: DAGCircuit) -> List[List[DAGOpNode]]:
    graph_layers = dag.multigraph_layers()
    try:
        next(graph_layers)  # Remove input nodes
    except StopIteration:
        return []

    layers = []
    for graph_layer in graph_layers:
        layer = [node for node in graph_layer if isinstance(node, DAGOpNode) and isinstance(node.op, Gate)]

        if not layer:
            return layers

        layers.append(layer)

    return layers


def op_node_at(dag: DAGCircuit, layer: int, qubit: int) -> Optional[DAGOpNode]:
    target_node = None
    layers = dag_layers(dag)

    if layer < len(layers):
        for op_node in layers[layer]:
            if qubit in {dag.qubits.index(q) for q in op_node.qargs}:
                target_node = op_node
                break

    return target_node
