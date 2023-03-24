
from math import pi

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qiskit_aer.noise import NoiseModel
from qiskit_ibm_provider import IBMProvider

from rich import print

from gradient_based import gradient_based_hst_weighted

if __name__ == '__main__':
    provider = IBMProvider()
    ibm_belem = provider.get_backend('ibmq_belem')

    noise_model = NoiseModel.from_backend(ibm_belem)

    u = QuantumCircuit(1)
    u.h(0)

    v = QuantumCircuit(1)
    v.rz(Parameter('a'), 0)
    v.sx(0)
    v.rz(Parameter('b'), 0)

    best_params, best_cost = gradient_based_hst_weighted(u, v, noise_model=noise_model)
    best_params_pi = [f'{p / pi:4f}π' for p in best_params]
    print(f'The best parameters were {best_params_pi} with a cost of {best_cost}.')
