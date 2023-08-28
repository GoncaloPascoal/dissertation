
import warnings
from argparse import ArgumentParser

from qiskit import QuantumCircuit
from tqdm import TqdmExperimentalWarning

from narlsqr.revlib import batch_convert_real_to_qasm
from narlsqr.utils import IBM_BASIS_GATES

def main():
    parser = ArgumentParser(
        'real_to_qasm',
        description='Converts RevLib circuit realizations (.real files) to OpenQASM',
    )

    parser.add_argument('src_dir', help='directory containing the .real files to convert')

    parser.add_argument('--dst-dir', metavar='P', help='directory where the OpenQASM files will be saved to')
    parser.add_argument('--min-qubits', metavar='N', type=int, default=0,
                        help='minimum circuit qubit count (inclusive)')
    parser.add_argument('--max-qubits', metavar='N', type=int, default=10_000,
                        help='maximum circuit qubit count (inclusive)')

    args = parser.parse_args()

    if args.min_qubits > args.max_qubits:
        raise ValueError('Minimum qubit count cannot be greater than maximum qubit count')

    def filter_fn(qc: QuantumCircuit) -> bool:
        return args.min_qubits <= qc.num_qubits <= args.max_qubits

    warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)

    batch_convert_real_to_qasm(
        args.src_dir,
        args.dst_dir,
        basis_gates=IBM_BASIS_GATES,
        filter_fn=filter_fn,
        use_tqdm=True,
    )


if __name__ == '__main__':
    main()
