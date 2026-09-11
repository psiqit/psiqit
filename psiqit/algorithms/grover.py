# psiqit/algorithms/grover.py 

"""
Grover's Search Algorithm
Unstructured search with quadratic speedup
"""
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, zero, one, plus, minus, basis
from ..quantum.operator import hadamard, pauli_x, pauli_z, cnot, toffoli
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# GROVER'S SEARCH ALGORITHM
# ============================================================================

def grover_search(
    n_qubits: int,
    target: int,
    n_iterations: Optional[int] = None,
    shots: int = 1024,
    oracle_type: str = 'phase',
    return_circuit: bool = False,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Grover's search algorithm for unstructured search
    
    Finds a target state in an unsorted database of size N = 2^n
    using O(√N) queries (quadratic speedup over classical O(N)).
    
    Args:
        n_qubits: Number of qubits (database size = 2^n)
        target: Target state to search for (0 to 2^n - 1)
        n_iterations: Number of Grover iterations (auto-calculated if None)
        shots: Number of measurement shots
        oracle_type: 'phase' or 'boolean' (phase oracle is more efficient)
        return_circuit: If True, return the circuit as well
        seed: Random seed for reproducibility
        
    Returns:
        Dict: Results including the most likely state and counts
        
    Example:
        >>> # Search for state |101⟩ in 3-qubit system
        >>> result = grover_search(n_qubits=3, target=5, shots=1024)
        >>> print(result['most_likely'])  # 5
        >>> print(result['success'])  # True if probability > 0.5
        
        >>> # Search with custom iterations
        >>> result = grover_search(n_qubits=4, target=10, n_iterations=3)
    """
    if seed is not None:
        np.random.seed(seed)
    
    validate_qubits(n_qubits)
    dim = 2 ** n_qubits
    
    if target < 0 or target >= dim:
        raise ValueError(f"Target {target} out of range [0, {dim-1}]")
    
    # Calculate optimal number of iterations
    if n_iterations is None:
        n_iterations = int(np.floor(np.pi / 4 * np.sqrt(dim)))
        logger.info(f"Auto-calculated {n_iterations} Grover iterations")
    
    logger.info(f"Grover search: {n_qubits} qubits, target={target}, iterations={n_iterations}")
    
    # Create circuit
    circuit = QuantumCircuit(n_qubits)
    
    # Initialize to |+⟩⊗n (superposition of all states)
    for i in range(n_qubits):
        circuit.h(i)
    
    # Grover iterations
    for iteration in range(n_iterations):
        # Oracle (marks target state)
        _apply_oracle(circuit, target, oracle_type=oracle_type)
        
        # Diffusion operator
        _apply_diffusion(circuit)
    
    # Measure
    result = circuit.measure(shots=shots)
    
    # Find most likely state
    counts = result['counts']
    most_likely_binary = max(counts, key=counts.get) if counts else None
    
    # Convert binary string to int
    most_likely_int = int(most_likely_binary, 2) if most_likely_binary else -1
    
    # Calculate probability of target
    target_binary = format(target, f'0{n_qubits}b')
    prob_target = counts.get(target_binary, 0) / shots if shots > 0 else 0
    
    # Determine success
    success = prob_target > 0.5
    
    # Build result
    result_dict = {
        'most_likely': most_likely_int,
        'most_likely_binary': most_likely_binary,
        'counts': counts,
        'shots': shots,
        'n_qubits': n_qubits,
        'target': target,
        'target_binary': target_binary,
        'iterations': n_iterations,
        'probability_target': prob_target,
        'success': success,
        'algorithm': 'grover'
    }
    
    if return_circuit:
        result_dict['circuit'] = circuit
    
    logger.info(f"Grover search completed: target={target}, found={most_likely_int}, success={success}")
    
    return result_dict


# ============================================================================
# GROVER WITH SIMULATED STATE VECTOR
# ============================================================================

def grover_search_simulated(
    n_qubits: int,
    target: int,
    n_iterations: Optional[int] = None
) -> Dict[str, Any]:
    """
    Simulated version of Grover's algorithm using state vectors
    
    This is faster and more accurate for testing, as it doesn't build circuits.
    
    Args:
        n_qubits: Number of qubits
        target: Target state to find
        n_iterations: Number of iterations (auto-calculated if None)
        
    Returns:
        Dict: Results with probabilities
        
    Example:
        >>> result = grover_search_simulated(n_qubits=3, target=5)
        >>> print(result['probability_target'])  # ~0.94
        >>> print(result['most_likely'])  # 5
    """
    validate_qubits(n_qubits)
    dim = 2 ** n_qubits
    
    if target < 0 or target >= dim:
        raise ValueError(f"Target {target} out of range [0, {dim-1}]")
    
    # Calculate optimal iterations
    if n_iterations is None:
        n_iterations = int(np.floor(np.pi / 4 * np.sqrt(dim)))
    
    # Initial state: |s⟩ = (1/√N) Σ|x⟩
    s = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Target state |w⟩
    w = np.zeros(dim, dtype=complex)
    w[target] = 1.0
    
    # Apply Grover iterations
    state = s.copy()
    for _ in range(n_iterations):
        # Reflect about |w⟩ (oracle)
        state = state - 2 * (np.vdot(w, state) * w)
        
        # Reflect about |s⟩ (diffusion)
        state = 2 * np.vdot(s, state) * s - state
    
    # Normalize
    state = state / np.linalg.norm(state)
    
    # Probabilities
    probs = np.abs(state) ** 2
    
    # Most likely state
    most_likely = int(np.argmax(probs))
    prob_target = float(probs[target])
    
    logger.info(f"Grover simulated: target={target}, prob={prob_target:.4f}")
    
    return {
        'most_likely': most_likely,
        'probability_target': prob_target,
        'probabilities': probs.tolist(),
        'target': target,
        'n_qubits': n_qubits,
        'iterations': n_iterations,
        'state': Ket(state),
        'algorithm': 'grover_simulated',
        'success': prob_target > 0.5
    }


# ============================================================================
# GROVER'S ORACLE
# ============================================================================

def _apply_oracle(circuit: QuantumCircuit, target: int, oracle_type: str = 'phase'):
    """
    Apply Grover's oracle to mark the target state
    
    Args:
        circuit: QuantumCircuit object
        target: Target state to mark
        oracle_type: 'phase' or 'boolean'
    """
    n_qubits = circuit.n_qubits
    
    if oracle_type == 'phase':
        # Phase oracle: flip the sign of the target state
        binary = format(target, f'0{n_qubits}b')
        
        # Apply X gates to flip qubits where target has 0
        for i, bit in enumerate(binary):
            if bit == '0':
                circuit.x(i)
        
        # Multi-controlled Z gate on all qubits
        if n_qubits == 1:
            circuit.z(0)
        elif n_qubits == 2:
            circuit.cz(0, 1)
        else:
            _apply_multi_controlled_z(circuit, list(range(n_qubits)))
        
        # Uncompute X gates
        for i, bit in enumerate(binary):
            if bit == '0':
                circuit.x(i)
    
    elif oracle_type == 'boolean':
        logger.warning("Boolean oracle not fully implemented, using phase oracle instead")
        _apply_oracle(circuit, target, oracle_type='phase')
    
    else:
        raise ValueError(f"Unknown oracle_type: {oracle_type}. Use 'phase' or 'boolean'")



def _apply_multi_controlled_z(circuit: QuantumCircuit, qubits: List[int]):
    """
    Apply a multi-controlled Z gate on the given qubits
    """
    n = len(qubits)
    
    if n == 1:
        circuit.z(qubits[0])
    elif n == 2:
        circuit.cz(qubits[0], qubits[1])
    elif n == 3:
        # CCZ decomposition using Toffoli: H(target) - CCX - H(target)
        # This is mathematically exact and uses existing gates
        circuit.h(qubits[2])
        circuit.toffoli(qubits[0], qubits[1], qubits[2])
        circuit.h(qubits[2])
    else:
        raise NotImplementedError("Multi-controlled Z for n > 3 requires ancilla qubits or advanced decomposition.")
# ============================================================================
# GROVER DIFFUSION
# ============================================================================

def _apply_diffusion(circuit: QuantumCircuit):
    """
    Apply Grover's diffusion operator
    
    The diffusion operator reflects about the average amplitude.
    
    Args:
        circuit: QuantumCircuit object
    """
    n = circuit.n_qubits
    
    # H ⊗ n
    for i in range(n):
        circuit.h(i)
    
    # X ⊗ n
    for i in range(n):
        circuit.x(i)
    
    # Multi-controlled Z on all qubits
    if n == 1:
        circuit.z(0)
    elif n == 2:
        circuit.cz(0, 1)
    else:
        _apply_multi_controlled_z(circuit, list(range(n)))
    
    # X ⊗ n
    for i in range(n):
        circuit.x(i)
    
    # H ⊗ n
    for i in range(n):
        circuit.h(i)

# ============================================================================
# GROVER WITH CUSTOM ORACLE
# ============================================================================

def grover_with_custom_oracle(
    n_qubits: int,
    oracle: Callable,
    n_iterations: Optional[int] = None,
    shots: int = 1024,
    return_circuit: bool = False
) -> Dict[str, Any]:
    """
    Grover's search with a custom oracle function
    
    Args:
        n_qubits: Number of qubits
        oracle: Oracle function that takes a circuit and qubits
        n_iterations: Number of Grover iterations
        shots: Number of measurement shots
        return_circuit: If True, return the circuit
        
    Returns:
        Dict: Results
        
    Example:
        >>> def my_oracle(circ, qubits):
        ...     # Mark state |101⟩
        ...     circ.x(qubits[0])
        ...     circ.x(qubits[2])
        ...     circ.cz(qubits[0], qubits[1])
        ...     circ.cz(qubits[1], qubits[2])
        ...     circ.x(qubits[0])
        ...     circ.x(qubits[2])
        >>> result = grover_with_custom_oracle(3, my_oracle)
    """
    validate_qubits(n_qubits)
    dim = 2 ** n_qubits
    
    # Calculate optimal iterations
    if n_iterations is None:
        n_iterations = int(np.floor(np.pi / 4 * np.sqrt(dim)))
    
    # Create circuit
    circuit = QuantumCircuit(n_qubits)
    
    # Initialize to |+⟩⊗n
    for i in range(n_qubits):
        circuit.h(i)
    
    # Grover iterations
    for _ in range(n_iterations):
        # Apply custom oracle
        oracle(circuit, list(range(n_qubits)))
        
        # Apply diffusion
        _apply_diffusion(circuit)
    
    # Measure
    result = circuit.measure(shots=shots)
    
    # Find most likely state
    counts = result['counts']
    most_likely_binary = max(counts, key=counts.get) if counts else None
    most_likely_int = int(most_likely_binary, 2) if most_likely_binary else -1
    
    result_dict = {
        'most_likely': most_likely_int,
        'most_likely_binary': most_likely_binary,
        'counts': counts,
        'shots': shots,
        'n_qubits': n_qubits,
        'iterations': n_iterations,
        'algorithm': 'grover_custom_oracle'
    }
    
    if return_circuit:
        result_dict['circuit'] = circuit
    
    return result_dict


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def optimal_grover_iterations(n_qubits: int) -> int:
    """
    Calculate the optimal number of Grover iterations
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        int: Optimal number of iterations
        
    Example:
        >>> optimal_grover_iterations(3)  # 2
        >>> optimal_grover_iterations(4)  # 3
    """
    dim = 2 ** n_qubits
    return int(np.floor(np.pi / 4 * np.sqrt(dim)))


def grover_probability(
    n_qubits: int,
    n_iterations: int,
    target: int = 0
) -> float:
    """
    Calculate the probability of finding the target state after n_iterations
    
    Args:
        n_qubits: Number of qubits
        n_iterations: Number of Grover iterations
        target: Target state (for reference)
        
    Returns:
        float: Probability of target state
        
    Example:
        >>> grover_probability(3, 2)  # ~0.945
    """
    dim = 2 ** n_qubits
    theta = 2 * np.arcsin(1 / np.sqrt(dim))
    prob = np.sin((2 * n_iterations + 1) * theta / 2) ** 2
    return float(prob)


def grover_amplitude(
    n_qubits: int,
    n_iterations: int,
    target: int = 0
) -> float:
    """
    Calculate the amplitude of the target state after n_iterations
    
    Args:
        n_qubits: Number of qubits
        n_iterations: Number of Grover iterations
        target: Target state (for reference)
        
    Returns:
        float: Amplitude of target state
        
    Example:
        >>> grover_amplitude(3, 2)  # ~0.972
    """
    return np.sqrt(grover_probability(n_qubits, n_iterations))


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'grover_search',
    'grover_search_simulated',
    'grover_with_custom_oracle',
    'optimal_grover_iterations',
    'grover_probability',
    'grover_amplitude',
]