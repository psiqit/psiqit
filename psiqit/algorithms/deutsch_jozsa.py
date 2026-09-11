#psiqit/algorithms/Deutsch-Jozsa.py
"""
Deutsch-Jozsa Algorithm
Determines if a function is constant or balanced
"""

import numpy as np
from typing import Optional, Dict, Any, List, Union, Callable
from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, zero, one, plus, minus
from ..quantum.operator import hadamard, pauli_x, pauli_z, cnot
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# DEUTSCH-JOZSA ALGORITHM
# ============================================================================

def deutsch_jozsa(
    oracle_function: Optional[Callable] = None,
    n_qubits: Optional[int] = None,
    function_type: Optional[str] = None,
    shots: int = 1024,
    return_circuit: bool = False
) -> Dict[str, Any]:
    """
    Deutsch-Jozsa algorithm to determine if a function is constant or balanced
    """
    logger.info("Running Deutsch-Jozsa algorithm")
    
    if n_qubits is None:
        if function_type is not None:
            n_qubits = 3
            logger.warning(f"n_qubits not provided, using default {n_qubits}")
        elif oracle_function is not None:
            n_qubits = 3
            logger.warning("n_qubits not provided, using default 3")
        else:
            raise ValueError("Either oracle_function or function_type must be provided")
    
    validate_qubits(n_qubits)
    
    if function_type is not None:
        oracle = _create_oracle_from_type(function_type, n_qubits)
    else:
        oracle = oracle_function
    
    circuit = QuantumCircuit(n_qubits + 1)
    
    circuit.x(n_qubits)
    circuit.h(n_qubits)
    
    for i in range(n_qubits):
        circuit.h(i)
    
    oracle(circuit, list(range(n_qubits)), n_qubits)
    
    for i in range(n_qubits):
        circuit.h(i)
    
    result = circuit.measure(shots=shots)
    
    # ==========================================================
    # اصلاح نهایی برای استخراج نتیجه (جایگزین کد قدیمی)
    # ==========================================================
    counts = result['counts']
    
    # کیوبیت داده‌ها در n_qubits بیت آخر (LSB) قرار دارند.
    # بیت اول (MSB) مربوط به Ancilla است که نباید شمرده شود.
    zero_state_data = '0' * n_qubits
    zero_count = 0
    
    for state, count in counts.items():
        if state[-n_qubits:] == zero_state_data:
            zero_count += count
            
    if zero_count / shots > 0.9:
        result_type = 'constant'
    else:
        result_type = 'balanced'
    # ==========================================================
    
    result_dict = {
        'result': result_type,
        'counts': counts,
        'shots': shots,
        'n_qubits': n_qubits,
        'function_type': function_type,
        'zero_count': zero_count,
        'zero_probability': zero_count / shots,
        'algorithm': 'deutsch_jozsa'
    }
    
    if return_circuit:
        result_dict['circuit'] = circuit
    
    logger.info(f"Deutsch-Jozsa result: {result_type}")
    
    return result_dict


# ============================================================================
# DEUTSCH ALGORITHM (1-QUBIT VERSION)
# ============================================================================

def deutsch_algorithm(
    oracle_function: Optional[Callable] = None,
    function_type: Optional[str] = None,
    shots: int = 1024,
    return_circuit: bool = False,
    n_qubits: int = 1
) -> Dict[str, Any]:
    """
    Deutsch algorithm (1-qubit version of Deutsch-Jozsa)
    """
    if function_type == 'identity' or function_type == 'not':
        function_type = 'balanced'
    
    return deutsch_jozsa(
        oracle_function=oracle_function,
        n_qubits=n_qubits,
        function_type=function_type,
        shots=shots,
        return_circuit=return_circuit
    )


# ============================================================================
# ORACLE CREATION
# ============================================================================

def _create_oracle_from_type(function_type: str, n_qubits: int) -> Callable:
    if function_type == 'constant_zero':
        def constant_zero_oracle(circ: QuantumCircuit, qubits: List[int], output: int):
            pass
        return constant_zero_oracle
    
    elif function_type == 'constant_one':
        def constant_one_oracle(circ: QuantumCircuit, qubits: List[int], output: int):
            circ.x(output)
        return constant_one_oracle
    
    elif function_type == 'balanced':
        def balanced_oracle(circ: QuantumCircuit, qubits: List[int], output: int):
            for q in qubits:
                circ.cx(q, output)
        return balanced_oracle
    
    else:
        raise ValueError(f"Unknown function_type: {function_type}")
    
def create_balanced_oracle(n_qubits: int) -> Callable:
    def balanced_oracle(circ: QuantumCircuit, qubits: List[int], output: int):
        for q in qubits:
            circ.cx(q, output)
    return balanced_oracle


def create_constant_oracle(n_qubits: int, value: int = 0) -> Callable:
    if value == 0:
        def constant_zero_oracle(circ: QuantumCircuit, qubits: List[int], output: int):
            pass
        return constant_zero_oracle
    else:
        def constant_one_oracle(circ: QuantumCircuit, qubits: List[int], output: int):
            circ.x(output)
        return constant_one_oracle


# ============================================================================
# SIMULATED DEUTSCH-JOZSA (State Vector)
# ============================================================================

def deutsch_jozsa_simulated(function_type: str, n_qubits: int = 3) -> Dict[str, Any]:
    if function_type in ['constant_zero', 'constant_one']:
        result_type = 'constant'
        zero_probability = 1.0
    elif function_type == 'balanced':
        result_type = 'balanced'
        zero_probability = 0.0
    else:
        raise ValueError(f"Unknown function_type: {function_type}")
    
    return {
        'result': result_type,
        'function_type': function_type,
        'n_qubits': n_qubits,
        'zero_probability': zero_probability,
        'algorithm': 'deutsch_jozsa_simulated'
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_constant_function(f: Callable[[List[int]], int], n_qubits: int) -> bool:
    first_value = None
    for i in range(2 ** n_qubits):
        x = [int(b) for b in format(i, f'0{n_qubits}b')]
        value = f(x)
        if first_value is None:
            first_value = value
        elif value != first_value:
            return False
    return True


def is_balanced_function(f: Callable[[List[int]], int], n_qubits: int) -> bool:
    count_zeros = 0
    count_ones = 0
    for i in range(2 ** n_qubits):
        x = [int(b) for b in format(i, f'0{n_qubits}b')]
        value = f(x)
        if value == 0:
            count_zeros += 1
        else:
            count_ones += 1
    return count_zeros == count_ones


def analyze_function_classically(f: Callable[[List[int]], int], n_qubits: int) -> str:
    if is_constant_function(f, n_qubits):
        return 'constant'
    elif is_balanced_function(f, n_qubits):
        return 'balanced'
    else:
        return 'neither'


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'deutsch_jozsa',
    'deutsch_algorithm',
    'create_balanced_oracle',
    'create_constant_oracle',
    'deutsch_jozsa_simulated',
    'is_constant_function',
    'is_balanced_function',
    'analyze_function_classically',
]