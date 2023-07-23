
import os
from collections.abc import Callable
from pathlib import Path
from typing import Optional, TypeAlias

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate, SwapGate, SXGate, SXdgGate, XGate
from tqdm.rich import tqdm


RealHeader: TypeAlias = dict[str, str | list[str]]
OperationList: TypeAlias = list[tuple[str, list[str]]]


# Peres gate
_peres_qc = QuantumCircuit(3)
_peres_qc.ccx(0, 1, 2)
_peres_qc.cx(0, 1)

peres_gate = _peres_qc.to_gate(label='PG')
peres_gate.name = 'peres'

del _peres_qc


def split_real_file(path: str) -> tuple[RealHeader, OperationList]:
    with open(path, 'r') as f:
        lines = [
            line.strip() for line in f.readlines()
            if not (line.isspace() or line.startswith('#'))
        ]

    header = {}
    ops = []

    for line in lines:
        if line.startswith('.'):
            parts = line[1:].split()
            key = parts[0]
            args = parts[1] if len(parts) == 2 else parts[1:]

            header[key] = args
        else:
            parts = line.split()
            name = parts[0]
            args = parts[1:]

            ops.append((name, args))

    return header, ops


def parse_real_file(path: str) -> QuantumCircuit:
    header, ops = split_real_file(path)

    num_qubits = int(header['numvars'])
    vars_to_qubits = {var: i for i, var in enumerate(header['variables'])}

    qc = QuantumCircuit(num_qubits)

    for name, args in ops:
        qargs = [vars_to_qubits[var] for var in args]
        gate_qubits = len(args)

        if name.startswith('t'):
            # Multi-controlled X gate
            op = MCXGate(gate_qubits - 1) if gate_qubits > 1 else XGate()
        elif name.startswith('f'):
            # Multi-controlled SWAP (Fredkin gate)
            op = SwapGate().control(gate_qubits - 2) if gate_qubits > 2 else SwapGate()
        elif name.startswith('p'):
            # Peres gate
            if gate_qubits != 3:
                raise NotImplementedError('Peres gate must apply to 3 qubits')
            op = peres_gate
        elif name.startswith('v+'):
            # Multi-controlled V+ (SXdg) gate
            op = SXdgGate().control(gate_qubits - 1) if gate_qubits > 1 else SXdgGate()
        else:
            # Multi-controlled V (SX) gate
            op = SXGate().control(gate_qubits - 1) if gate_qubits > 1 else SXGate()

        qc.append(op, qargs)

    return qc


def convert_real_to_qasm(src: str, dst_dir: Optional[str] = None, *, basis_gates: Optional[list[str]] = None):
    """
    Converts a ``.real`` file at :py:attr:`src` to OpenQASM, storing it in the :py:attr:`dst_dir` directory.
    :param src: path to the ``.real`` file.
    :param dst_dir: path to the directory where the ``.qasm`` file should be saved. If not specified, it will
        correspond to the current working directory.
    :param basis_gates: if not ``None``, decompose the circuit using the given set of basis gates.
    """
    dst_dir = os.getcwd() if dst_dir is None else dst_dir
    file_name = Path(src).stem

    qc = parse_real_file(src)
    if basis_gates is not None:
        qc = transpile(qc, basis_gates=basis_gates, optimization_level=0)
    qc.qasm(filename=os.path.join(dst_dir, f'{file_name}.qasm'))


def batch_convert_real_to_qasm(
    src_dir: str,
    dst_dir: Optional[str] = None,
    *,
    basis_gates: Optional[list[str]] = None,
    condition: Optional[Callable[[str], bool]] = None,
    use_tqdm: bool = False,
):
    files = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]
    files = [f for f in files if os.path.isfile(f)]

    if use_tqdm:
        files = tqdm(files)

    for file in files:
        if condition is None or condition(file):
            convert_real_to_qasm(file, dst_dir, basis_gates=basis_gates)
