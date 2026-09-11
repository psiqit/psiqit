"""
PSIQIT - Python Scientific Quantum Information Toolkit
Quantum computing, quantum information theory, and quantum machine learning
Version 1.0.3
"""

# Import version
from .version import __version__

# Core modules
from .circuits import QuantumCircuit, QuantumRegister, Qubit
from .quantum import Ket, Bra, Operator
from .algorithms import grover_search, shor_algorithm, qft
from .variational import VQE, QAOA
from .info import entropy, concurrence, negativity
from .visualization import bloch_sphere, draw_circuit


__all__ = [
    '__version__',
    'QuantumCircuit',
    'QuantumRegister',
    'Qubit',
    'Ket',
    'Bra',
    'Operator',
    'grover_search',
    'shor_algorithm',
    'qft',
    'VQE',
    'QAOA',
    'entropy',
    'concurrence',
    'negativity',
    'bloch_sphere',
    'draw_circuit',
    'PetuniaAssistant',
]

__author__ = "Mahdi Azadmarzabadi"
__email__ = "psiqitofficial@protonmail.com"
__description__ = "Python Scientific Quantum Information Toolkit"
__status__ = "Development"