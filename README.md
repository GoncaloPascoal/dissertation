
<div align="center">
  <h1>Noise-Adaptive Reinforcement Learning Strategies for Qubit Routing</h1>
  <h2>Master's Dissertation</h2>
  <h3>Faculdade de Engenharia da Universidade do Porto</h3>

  Gonçalo Pascoal<br>
  up201806332@edu.fe.up.pt
</div>

## Overview

This dissertation proposes a method to address the **qubit routing problem** using deep reinforcement learning, specifically using **Proximal Policy Optimization** (PPO). Existing quantum computers are constrained by several factors. In addition to having a small number of qubits (quantum bits) and being highly susceptible to errors, current devices only support a narrow set of operations and have sparse qubit connectivity, preventing interactions between arbitrary qubits. Therefore, quantum algorithms must be compiled in order to comply with the constraints of the target computer.

Qubit routing, one of the most crucial stages in the compilation process, introduces additional operations to ensure that multi-qubit gates are only performed between connected qubits. It is a challenging combinatorial optimization problem that has been shown to be NP-complete.

Our RL environment represents quantum circuits as numerical matrices, and we also provide the error rates of individual device edges to the models. Possible actions include inserting SWAP or BRIDGE operations, and we use action masking to prevent invalid actions and speed up training. To further optimize the routing process, we also perform gate commutation analysis. We have evaluated this approach for five IBM quantum computers ranging from 5 to 27 qubits, using both randomly generated and realistic circuits.

## Instructions

This project requires **Python 3.11** or higher. To install the necessary dependencies, run the following command on the project's root directory:

```bash
pip install -r requirements.txt
```

We recommend using a Python virtual environment (`venv`) to avoid conflicts with other packages that may already be installed.

A set of utility command-line scripts is provided within the `scripts` folder. These scripts should generally be executed as a module (using the `-m` command-line option of the Python interpreter) and from the root project directory:

```bash
python3 -m scripts.train [...]
```

Below is a short description of each of the scripts. Many of these support command-line arguments (use `-h` to display the available options).

- `train.py`: used for training the RL models. User must provide paths to environment and training configuration files (see the examples in the `config` folder);
- `eval.py`: evaluate a model's performance according to several metrics, saving the results to a file (using `pickle`);
- `real_to_qasm.py`: converts circuits defined using RevLib's `.real` file format to OpenQASM;
- `analyze_results.py`: generates the plots and data used in the Chapter 6 of the dissertation (*Results and Discussion*);
- `ibmq/register_account.py`: locally store an IBM quantum API key so that information about IBM devices can be fetched using the `qiskit_ibm_provider` package;
- `ibmq/current_calibration.py`: fetches the latest calibration data from one or more IBM devices, storing it as a JSON file.
