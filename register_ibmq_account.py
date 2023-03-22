
from qiskit_ibm_provider import IBMProvider
from getpass import getpass

token = getpass('Enter an IBM Quantum API token: ')
IBMProvider.save_account(token)
