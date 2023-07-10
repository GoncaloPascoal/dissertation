
from getpass import getpass

from qiskit_ibm_provider import IBMProvider


if __name__ == '__main__':
    token = getpass('Enter an IBM Quantum API token: ')
    IBMProvider.save_account(token)
