
from getpass import getpass

from qiskit_ibm_runtime import QiskitRuntimeService


if __name__ == '__main__':
    token = getpass('Enter an IBM Quantum API token: ')
    QiskitRuntimeService.save_account(
        token,
        instance='ibm_quantum',
        overwrite=True,
        set_as_default=True,
    )
