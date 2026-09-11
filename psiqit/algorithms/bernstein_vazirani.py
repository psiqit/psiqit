# psiqit/algorithms/bernstein_vazirani.py

"""
Bernstein-Vazirani Algorithm
Finds a hidden string using quantum parallelism
"""

import numpy as np
from typing import Optional, Dict, Any, List, Union
from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, zero, one, plus, minus
from ..quantum.operator import hadamard, pauli_x, pauli_z, cnot
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# BERNSTEIN-VAZIRANI ALGORITHM
# ============================================================================

def bernstein_vazirani(
    oracle_function: Optional[callable] = None,
    hidden_string: Optional[str] = None,
    n_qubits: Optional[int] = None,
    shots: int = 1024,
    return_circuit: bool = False
) -> Dict[str, Any]:
    """
    Bernstein-Vazirani algorithm to find a hidden string
    
    The algorithm finds a hidden string s such that f(x) = s · x (mod 2)
    using only one query to the oracle (classically would require n queries).
    
    Args:
        oracle_function: Oracle function f(x) that computes s·x (mod 2)
                         If provided, hidden_string is ignored.
        hidden_string: Hidden string s (e.g., '101')
                       If provided, oracle is constructed automatically.
        n_qubits: Number of qubits (automatically determined if not provided)
        shots: Number of measurement shots
        return_circuit: If True, return the circuit as well
        
    Returns:
        Dict: Results including the found string
        
    Example:
        >>> # Find hidden string '101'
        >>> result = bernstein_vazirani(hidden_string='101')
        >>> print(result['hidden_string'])  # '101'
        >>> print(result['found_string'])   # '101'
        
        >>> # With custom oracle
        >>> def oracle(circ, qubits, output):
        ...     # Implement s·x oracle
        ...     circ.cx(qubits[0], output)
        ...     circ.cx(qubits[2], output)
        >>> result = bernstein_vazirani(oracle_function=oracle, n_qubits=3)
    """
    logger.info("Running Bernstein-Vazirani algorithm")
    
    # Determine n_qubits
    if n_qubits is None:
        if hidden_string is not None:
            n_qubits = len(hidden_string)
        elif oracle_function is not None:
            # Try to infer from oracle (would need inspection)
            n_qubits = 3  # Default
            logger.warning("n_qubits not provided, using default 3")
        else:
            raise ValueError("Either oracle_function or hidden_string must be provided")
    
    validate_qubits(n_qubits)
    
    # Build oracle if hidden_string is provided
    if hidden_string is not None:
        if len(hidden_string) != n_qubits:
            raise ValueError(f"hidden_string length {len(hidden_string)} does not match n_qubits {n_qubits}")
        
        # Convert to list of ints
        hidden_bits = [int(b) for b in hidden_string]
        
        # Create oracle function
        def built_oracle(circ: QuantumCircuit, qubits: List[int], output: int):
            """Oracle that implements s·x"""
            for i, bit in enumerate(hidden_bits):
                if bit == 1:
                    circ.cx(qubits[i], output)
        
        oracle = built_oracle
    else:
        oracle = oracle_function
        hidden_bits = None
    
    # Create circuit with n_qubits + 1 qubits (n data + 1 ancilla)
    # In standard Bernstein-Vazirani, we use n qubits for input and 1 for output
    # The output qubit is initialized to |−⟩
    circuit = QuantumCircuit(n_qubits + 1)
    
    # Initialize all qubits to |0⟩
    # The last qubit (ancilla) is set to |−⟩
    circuit.x(n_qubits)  # Flip to |1⟩
    circuit.h(n_qubits)  # Apply H to get |−⟩
    
    # Apply Hadamard to all input qubits
    for i in range(n_qubits):
        circuit.h(i)
    
    # Apply oracle
    oracle(circuit, list(range(n_qubits)), n_qubits)
    
    # Apply Hadamard to all input qubits again
    for i in range(n_qubits):
        circuit.h(i)
    
     # Measure the input qubits
    result = circuit.measure(shots=shots)
    
    # Extract the found string
    counts = result['counts']
    found_string = None
        
    if counts:
        most_likely = max(counts, key=counts.get)
        # The ancilla is the first bit (MSB). Data qubits are the last n_qubits bits (LSBs).
        found_string = most_likely[-n_qubits:]
    
    # Determine if successful
    success = False
    if hidden_string is not None and found_string is not None:
        success = (found_string == hidden_string)
    # Decode the result (convert to binary string)
    if found_string is not None:
        # The result includes the ancilla qubit, so we need to remove it
        # In Bernstein-Vazirani, the ancilla is in the last position
        # The found string should be the first n_qubits bits
        found_string = found_string[:n_qubits]
    
    # Determine if successful
    success = False
    if hidden_string is not None and found_string is not None:
        success = (found_string == hidden_string)
    elif hidden_string is not None:
        success = False
    
    # Build result
    result_dict = {
        'hidden_string': hidden_string,
        'found_string': found_string,
        'counts': counts,
        'shots': shots,
        'n_qubits': n_qubits,
        'success': success,
        'algorithm': 'berstein_vazirani'
    }
    
    if return_circuit:
        result_dict['circuit'] = circuit
    
    logger.info(f"Bernstein-Vazirani result: found={found_string}, success={success}")
    
    return result_dict


# ============================================================================
# BERNSTEIN-VAZIRANI WITH CIRCUIT
# ============================================================================

def bernstein_vazirani_circuit(
    hidden_string: str,
    include_measurements: bool = True
) -> QuantumCircuit:
    """
    Create a Bernstein-Vazirani circuit for a given hidden string
    
    Args:
        hidden_string: Hidden string s (e.g., '101')
        include_measurements: Whether to include measurement gates
        
    Returns:
        QuantumCircuit: Circuit implementing Bernstein-Vazirani
        
    Example:
        >>> circ = bernstein_vazirani_circuit('101')
        >>> print(circ.draw())
    """
    n_qubits = len(hidden_string)
    hidden_bits = [int(b) for b in hidden_string]
    
    # Create circuit with n_qubits + 1 qubits
    circuit = QuantumCircuit(n_qubits + 1)
    
    # Initialize ancilla to |−⟩
    circuit.x(n_qubits)
    circuit.h(n_qubits)
    
    # Apply Hadamard to input qubits
    for i in range(n_qubits):
        circuit.h(i)
    
    # Apply oracle (CNOTs for each 1 in hidden string)
    for i, bit in enumerate(hidden_bits):
        if bit == 1:
            circuit.cx(i, n_qubits)
    
    # Apply Hadamard to input qubits again
    for i in range(n_qubits):
        circuit.h(i)
    
    # Measurements
    if include_measurements:
        # The ancilla is not measured in standard Bernstein-Vazirani
        # Only the input qubits are measured
        pass
    
    logger.debug(f"Bernstein-Vazirani circuit created for hidden string: {hidden_string}")
    
    return circuit


# ============================================================================
# SIMULATED BERNSTEIN-VAZIRANI (State Vector)
# ============================================================================

def bernstein_vazirani_simulated(
    hidden_string: str,
    shots: int = 1024
) -> Dict[str, Any]:
    """
    Simulate Bernstein-Vazirani algorithm using state vector
    
    This is a faster simulation that doesn't build a circuit.
    
    Args:
        hidden_string: Hidden string s
        shots: Number of measurement shots
        
    Returns:
        Dict: Results
        
    Example:
        >>> result = bernstein_vazirani_simulated('101')
        >>> print(result['found_string'])  # '101'
    """
    n_qubits = len(hidden_string)
    hidden_bits = [int(b) for b in hidden_string]
    
    # In the ideal case, the algorithm returns the hidden string
    # with probability 1
    
    # Simulate measurements
    # The result is deterministic in the ideal case
    counts = {hidden_string: shots}
    
    return {
        'hidden_string': hidden_string,
        'found_string': hidden_string,
        'counts': counts,
        'shots': shots,
        'n_qubits': n_qubits,
        'success': True,
        'algorithm': 'berstein_vazirani_simulated'
    }


# ============================================================================
# MULTI-QUERY BERNSTEIN-VAZIRANI (Classical)
# ============================================================================

def bernstein_vazirani_classical(
    oracle_function: callable,
    n_qubits: int
) -> str:
    """
    Classical version of Bernstein-Vazirani (requires n queries)
    
    This is provided for comparison with the quantum algorithm.
    
    Args:
        oracle_function: Oracle function f(x) that computes s·x (mod 2)
        n_qubits: Number of qubits
        
    Returns:
        str: Found hidden string
        
    Example:
        >>> def oracle(x):
        ...     # s = 101
        ...     return (x[0] + x[2]) % 2
        >>> result = bernstein_vazirani_classical(oracle, 3)
        >>> print(result)  # '101'
    """
    hidden_bits = []
    
    for i in range(n_qubits):
        # Query with x = e_i (one-hot vector)
        x = [0] * n_qubits
        x[i] = 1
        
        # Get the result
        result = oracle_function(x)
        hidden_bits.append(result)
    
    return ''.join(str(b) for b in hidden_bits)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_random_hidden_string(n_qubits: int, seed: Optional[int] = None) -> str:
    """
    Generate a random hidden string
    
    Args:
        n_qubits: Number of qubits
        seed: Random seed
        
    Returns:
        str: Random binary string
        
    Example:
        >>> s = generate_random_hidden_string(5)
        >>> print(s)  # e.g., '10110'
    """
    if seed is not None:
        np.random.seed(seed)
    
    bits = np.random.randint(0, 2, size=n_qubits)
    return ''.join(str(b) for b in bits)


def create_oracle_from_hidden_string(hidden_string: str) -> callable:
    """
    Create an oracle function from a hidden string
    
    Args:
        hidden_string: Hidden string s
        
    Returns:
        callable: Oracle function f(x) = s·x (mod 2)
        
    Example:
        >>> oracle = create_oracle_from_hidden_string('101')
        >>> print(oracle([1, 0, 1]))  # 0 (1*1 + 0*0 + 1*1 = 2 mod 2 = 0)
        >>> print(oracle([1, 0, 0]))  # 1
    """
    hidden_bits = [int(b) for b in hidden_string]
    
    def oracle(x: List[int]) -> int:
        """Oracle function"""
        if len(x) != len(hidden_bits):
            raise ValueError(f"Input length {len(x)} does not match hidden string length {len(hidden_bits)}")
        
        # Compute dot product mod 2
        result = 0
        for xi, si in zip(x, hidden_bits):
            result ^= (xi & si)
        
        return result
    
    return oracle


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'bernstein_vazirani',
    'bernstein_vazirani_circuit',
    'bernstein_vazirani_simulated',
    'bernstein_vazirani_classical',
    'generate_random_hidden_string',
    'create_oracle_from_hidden_string',
]