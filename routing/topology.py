
import rustworkx as rx


def t_topology() -> rx.PyGraph:
    """
    T-shape topology present in certain 5-qubit IBM devices (``ibmq_{belem, lima, quito}``).
    """
    g = rx.PyGraph()
    g.extend_from_edge_list([(0, 1), (1, 2), (1, 3), (3, 4)])
    return g

def h_topology() -> rx.PyGraph:
    """
    H-shape topology present in certain 7-qubit IBM devices (``ibmq_{jakarta, lagos, nairobi, perth}``).
    """
    g = rx.PyGraph()
    g.extend_from_edge_list([(0, 1), (1, 2), (1, 3), (3, 5), (4, 5), (5, 6)])
    return g

def grid_topology(rows: int, cols: int) -> rx.PyGraph:
    """
    Two-dimensional grid topology. Qubit count is equal to :py:attr:`rows` x :py:attr:`cols`.
    :param rows: number of rows in the grid.
    :param cols: number of columns in the grid.
    """
    g = rx.PyGraph()

    g.add_nodes_from(range(rows * cols))

    for row in range(rows):
        for col in range(cols):
            if col != cols - 1:
                g.add_edge(row * cols + col, row * cols + col + 1, None)

            if row != rows - 1:
                g.add_edge(row * cols + col, (row + 1) * cols + col, None)

    return g

def linear_topology(num_qubits: int) -> rx.PyGraph:
    """
    Linear nearest-neighbor topology. Present in the 5-qubit ``ibmq_manila`` IBM device.
    :param num_qubits: the device's qubit count.
    """
    g = rx.PyGraph()
    g.extend_from_edge_list([(i, i + 1) for i in range(num_qubits - 1)])
    return g

def ibm_16q_topology() -> rx.PyGraph:
    """
    Topology used by 16-qubit IBM Falcon chips, such as the ``ibmq_guadalupe`` device.
    """
    g = rx.PyGraph()
    g.extend_from_edge_list([
        (0, 1), (1, 2), (1, 4), (2, 3), (3, 5), (4, 7), (5, 8), (6, 7),
        (7, 10), (8, 9), (8, 11), (10, 12), (11, 14), (12, 13), (12, 15), (13, 14),
    ])
    return g

def ibm_27q_topology() -> rx.PyGraph:
    """
    Topology used by 27-qubit IBM Falcon chips, such as the ``ibmq_mumbai`` device.
    """
    g = rx.PyGraph()
    g.extend_from_edge_list([
        (0, 1), (1, 2), (1, 4), (2, 3), (3, 5), (4, 7), (5, 8), (6, 7),
        (7, 10), (8, 9), (8, 11), (10, 12), (11, 14), (12, 13), (12, 15), (13, 14),
        (14, 16), (15, 18), (16, 19), (17, 18), (18, 21), (19, 20), (19, 22), (21, 23),
        (22, 25), (23, 24), (24, 25), (25, 26),
    ])
    return g
