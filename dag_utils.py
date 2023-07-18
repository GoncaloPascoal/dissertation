
from typing import List

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
