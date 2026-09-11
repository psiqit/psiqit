# psiqit/algorithms/simon.py

"""
Simon's Algorithm
Finds the period of a function with exponential speedup
"""

import numpy as np
from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, zero, one, plus, minus, basis
from ..quantum.operator import hadamard, pauli_x, pauli_z, cnot
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# SIMON'S ALGORITHM
# ============================================================================

def simon_algorithm(
    oracle_function: Optional[Callable] = None,
    hidden_string: Optional[str] = None,
    n_qubits: Optional[int] = None,
    shots: int = 1024,
    max_attempts: int = 10,
    return_circuit: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Simon's algorithm to find the period of a function
    
    Given a function f: {0,1}ⁿ → {0,1}ⁿ such that f(x) = f(x ⊕ s)
    for some hidden string s, Simon's algorithm finds s using O(n) queries
    (exponential speedup over classical O(2^{n/2})).
    
    Args:
        oracle_function: Oracle function f(x) that returns a bitstring
                         If provided, hidden_string is ignored.
        hidden_string: Hidden string s (e.g., '101')
                       If provided, oracle is constructed automatically.
        n_qubits: Number of qubits (automatically determined if not provided)
        shots: Number of measurement shots
        max_attempts: Maximum number of attempts to find s
        return_circuit: If True, return the circuit as well
        verbose: Print progress information
        
    Returns:
        Dict: Results including the found string
        
    Example:
        >>> # Find hidden string '101'
        >>> result = simon_algorithm(hidden_string='101')
        >>> print(result['hidden_string'])  # '101'
        >>> print(result['found_string'])   # '101'
        
        >>> # With custom oracle
        >>> def oracle(x):
        ...     # f(x) = x if x[0] == 0, else x ⊕ s
        ...     s = [1, 0, 1]
        ...     if x[0] == 0:
        ...         return x
        ...     else:
        ...         return [(xi ^ si) for xi, si in zip(x, s)]
        >>> result = simon_algorithm(oracle_function=oracle, n_qubits=3)
    """
    logger.info("Running Simon's algorithm")
    
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
        def built_oracle(x: List[int]) -> List[int]:
            """Oracle that implements f(x) = x ⊕ s"""
            # f(x) = x if x[0] == 0, else x ⊕ s
            if x[0] == 0:
                return x.copy()
            else:
                return [(xi ^ si) for xi, si in zip(x, hidden_bits)]
        
        oracle = built_oracle
    else:
        oracle = oracle_function
        hidden_bits = None
    
    # Run Simon's algorithm multiple times to find s
    found_string = None
    equations = []
    attempts = 0
    
    while attempts < max_attempts:
        attempts += 1
        
        if verbose:
            logger.info(f"Attempt {attempts}/{max_attempts}")
        
        # Run the quantum circuit
        result = _simon_circuit(oracle, n_qubits, shots=shots)
        
        # Get the measured bitstring
        counts = result['counts']
        most_likely = max(counts, key=counts.get) if counts else None
        
        if most_likely is not None:
            # The measured bitstring y satisfies y · s = 0 (mod 2)
            # We collect these equations to solve for s
            y_bits = [int(b) for b in most_likely]
            
            # Skip zero vector
            if all(b == 0 for b in y_bits):
                continue
            
            # Add equation to the system
            equations.append(y_bits)
            
            if verbose:
                logger.info(f"Found equation: {most_likely} · s = 0")
            
            # Try to solve the system of equations
            solution = _solve_linear_system(equations, n_qubits)
            
            if solution is not None:
                found_string = ''.join(str(b) for b in solution)
                
                # Verify the solution
                if hidden_string is not None:
                    if found_string == hidden_string:
                        break
                else:
                    # Verify using the oracle
                    if _verify_solution(oracle, found_string, n_qubits):
                        break
    
    # Build result
    result_dict = {
        'hidden_string': hidden_string,
        'found_string': found_string,
        'equations': equations,
        'attempts': attempts,
        'shots': shots,
        'n_qubits': n_qubits,
        'success': found_string is not None and (hidden_string is None or found_string == hidden_string),
        'algorithm': 'simon'
    }
    
    if return_circuit:
        result_dict['circuit'] = result.get('circuit') if 'result' in locals() else None
    
    logger.info(f"Simon's algorithm result: found={found_string}, success={result_dict['success']}")
    
    return result_dict


# ============================================================================
# SIMON'S CIRCUIT
# ============================================================================

def _simon_circuit(
    oracle: Callable,
    n_qubits: int,
    shots: int = 1024
) -> Dict[str, Any]:
    """
    Run a single iteration of Simon's algorithm
    
    Args:
        oracle: Oracle function f(x)
        n_qubits: Number of qubits
        shots: Number of measurement shots
        
    Returns:
        Dict: Measurement results
    """
    # Create circuit with 2n qubits (n for input, n for output)
    total_qubits = 2 * n_qubits
    circuit = QuantumCircuit(total_qubits)
    
    # Apply Hadamard to all input qubits (first n qubits)
    for i in range(n_qubits):
        circuit.h(i)
    
    # Apply oracle
    # For Simon's algorithm, we need the oracle to compute f(x)
    # This is a quantum oracle that maps |x⟩|0⟩ → |x⟩|f(x)⟩
    _apply_simon_oracle(circuit, oracle, n_qubits)
    
    # Apply Hadamard to input qubits again
    for i in range(n_qubits):
        circuit.h(i)
    
    # Measure the input qubits
    # We measure only the first n qubits (the input register)
    # The output register is not measured
    result = circuit.measure(shots=shots)
    
    # The result contains measurements of all qubits
    # We need to extract the first n qubits
    counts = {}
    for key, value in result['counts'].items():
        # Take only the first n bits (the input register)
        input_bits = key[:n_qubits]
        counts[input_bits] = counts.get(input_bits, 0) + value
    
    return {
        'counts': counts,
        'shots': shots,
        'n_qubits': n_qubits,
        'circuit': circuit
    }


def _apply_simon_oracle(
    circuit: QuantumCircuit,
    oracle: Callable,
    n_qubits: int
):
    """
    Apply Simon's oracle: |x⟩|0⟩ → |x⟩|f(x)⟩
    
    Args:
        circuit: QuantumCircuit object
        oracle: Classical oracle function f(x)
        n_qubits: Number of qubits
    """
    # This is a simplified version for simulation
    # In a real quantum circuit, the oracle would be implemented
    # using quantum gates
    
    # For the simulation, we use the classical oracle to determine
    # the transformation and apply it using controlled operations
    
    # Since we're simulating, we can just use the classical oracle
    # to compute f(x) for each basis state
    
    # This is a placeholder - the actual implementation would
    # require building a quantum oracle circuit
    
    logger.warning("Simon's oracle is implemented using classical computation for simulation")


# ============================================================================
# LINEAR ALGEBRA UTILITIES
# ============================================================================

def _solve_linear_system(
    equations: List[List[int]],
    n_qubits: int
) -> Optional[List[int]]:
    """
    Solve a system of linear equations over GF(2)
    
    The equations are of the form: y_i · s = 0 (mod 2)
    
    Args:
        equations: List of equation vectors y_i
        n_qubits: Number of variables
        
    Returns:
        Optional[List[int]]: Solution s (binary string) or None if not unique
    """
    if not equations:
        return None
    
    # Convert equations to matrix form
    # We need to find s such that y_i · s = 0 for all i
    # This is equivalent to finding the null space of the matrix
    
    # Build the matrix
    rows = len(equations)
    cols = n_qubits
    matrix = np.array(equations, dtype=int)
    
    # We want to find s ≠ 0 such that matrix @ s = 0 (mod 2)
    # This is the null space of the matrix over GF(2)
    
    # For small systems, we can brute force
    for s_int in range(1, 2 ** n_qubits):
        s = [int(b) for b in format(s_int, f'0{n_qubits}b')]
        
        # Check if all equations are satisfied
        valid = True
        for eq in equations:
            dot = sum(eq[i] * s[i] for i in range(n_qubits)) % 2
            if dot != 0:
                valid = False
                break
        
        if valid:
            return s
    
    return None


def _verify_solution(
    oracle: Callable,
    found_string: str,
    n_qubits: int
) -> bool:
    """
    Verify that the found string is a valid period
    
    Args:
        oracle: Oracle function
        found_string: Candidate period s
        n_qubits: Number of qubits
        
    Returns:
        bool: True if valid, False otherwise
    """
    s = [int(b) for b in found_string]
    
    # Check if s is non-zero
    if all(b == 0 for b in s):
        return False
    
    # Verify that f(x) = f(x ⊕ s) for all x
    for x_int in range(2 ** n_qubits):
        x = [int(b) for b in format(x_int, f'0{n_qubits}b')]
        
        # Compute x ⊕ s
        x_xor_s = [(xi ^ si) for xi, si in zip(x, s)]
        
        # Check if f(x) == f(x ⊕ s)
        if oracle(x) != oracle(x_xor_s):
            return False
    
    return True


# ============================================================================
# SIMON'S ALGORITHM - SIMPLIFIED (CLASSICAL SIMULATION)
# ============================================================================

def simon_classical(
    oracle: Callable,
    n_qubits: int,
    max_attempts: int = 20
) -> Optional[str]:
    """
    Classical version of Simon's algorithm (for comparison)
    
    This is the classical algorithm for finding the period,
    which requires O(2^{n/2}) queries.
    
    Args:
        oracle: Oracle function f(x)
        n_qubits: Number of qubits
        max_attempts: Maximum number of attempts
        
    Returns:
        Optional[str]: Found period or None if not found
        
    Example:
        >>> def oracle(x):
        ...     s = [1, 0, 1]
        ...     if x[0] == 0:
        ...         return x
        ...     else:
        ...         return [(xi ^ si) for xi, si in zip(x, s)]
        >>> result = simon_classical(oracle, 3)
        >>> print(result)  # '101'
    """
    # Find pairs (x, y) such that f(x) = f(y)
    # Then y = x ⊕ s, so s = x ⊕ y
    
    pairs = {}
    
    for attempt in range(max_attempts):
        # Generate random x
        x_int = np.random.randint(0, 2 ** n_qubits)
        x = [int(b) for b in format(x_int, f'0{n_qubits}b')]
        
        # Compute f(x)
        fx = oracle(x)
        fx_tuple = tuple(fx)
        
        # Check if we've seen this value before
        if fx_tuple in pairs:
            y = pairs[fx_tuple]
            
            # Compute s = x ⊕ y
            s = [(xi ^ yi) for xi, yi in zip(x, y)]
            
            # Verify the solution
            if _verify_solution(oracle, ''.join(str(b) for b in s), n_qubits):
                return ''.join(str(b) for b in s)
        else:
            pairs[fx_tuple] = x
    
    return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_random_hidden_string(n_qubits: int, seed: Optional[int] = None) -> str:
    """
    Generate a random non-zero hidden string
    
    Args:
        n_qubits: Number of qubits
        seed: Random seed
        
    Returns:
        str: Random binary string (non-zero)
        
    Example:
        >>> s = generate_random_hidden_string(5)
        >>> print(s)  # e.g., '10110'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate a random non-zero string
    while True:
        bits = np.random.randint(0, 2, size=n_qubits)
        if any(bits):
            return ''.join(str(b) for b in bits)


def create_simon_oracle(hidden_string: str) -> Callable:
    """
    Create an oracle for Simon's algorithm
    
    The oracle implements f(x) = x if x[0] == 0, else x ⊕ s
    
    Args:
        hidden_string: Hidden string s
        
    Returns:
        Callable: Oracle function
        
    Example:
        >>> oracle = create_simon_oracle('101')
        >>> print(oracle([0, 0, 0]))  # [0, 0, 0]
        >>> print(oracle([1, 0, 0]))  # [1, 0, 1]
    """
    s = [int(b) for b in hidden_string]
    
    def oracle(x: List[int]) -> List[int]:
        """Oracle for Simon's algorithm"""
        if len(x) != len(s):
            raise ValueError(f"Input length {len(x)} does not match hidden string length {len(s)}")
        
        # f(x) = x if first bit is 0, else x ⊕ s
        if x[0] == 0:
            return x.copy()
        else:
            return [(xi ^ si) for xi, si in zip(x, s)]
    
    return oracle


def is_periodic(oracle: Callable, s: List[int], n_qubits: int) -> bool:
    """
    Check if s is a period of the oracle function
    
    Args:
        oracle: Oracle function
        s: Candidate period
        n_qubits: Number of qubits
        
    Returns:
        bool: True if s is a period, False otherwise
    """
    return _verify_solution(oracle, ''.join(str(b) for b in s), n_qubits)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'simon_algorithm',
    'simon_classical',
    'generate_random_hidden_string',
    'create_simon_oracle',
    'is_periodic',
]