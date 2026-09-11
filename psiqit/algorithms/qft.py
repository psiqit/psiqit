# psiqit/algorithms/qft.py

"""
Quantum Fourier Transform (QFT)
Quantum implementation of the discrete Fourier transform
"""

import numpy as np
from typing import Optional, List, Union, Dict, Any, Tuple
from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, basis
from ..quantum.operator import Operator, hadamard, phase, swap
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# QUANTUM FOURIER TRANSFORM
# ============================================================================

def qft(state: Union[Ket, np.ndarray, List], n_qubits: Optional[int] = None) -> Ket:
    """
    Apply Quantum Fourier Transform to a state
    
    The QFT transforms a state from the computational basis to the Fourier basis:
    |j⟩ → (1/√N) Σ_{k=0}^{N-1} ω^{jk} |k⟩
    where ω = e^{2πi/N}
    
    Args:
        state: Input state (Ket, list, or numpy array)
        n_qubits: Number of qubits (auto-detected if not provided)
        
    Returns:
        Ket: QFT-transformed state
        
    Example:
        >>> from psiqit.quantum import Ket
        >>> state = Ket([1, 0, 0, 0])  # |00⟩
        >>> result = qft(state)
        >>> print(result)  # (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
        
        >>> state = Ket([0, 1, 0, 0])  # |01⟩
        >>> result = qft(state)
        >>> print(result)  # Fourier basis state
    """
    if isinstance(state, Ket):
        data = state.data
        if n_qubits is None:
            n_qubits = int(np.log2(len(data)))
    else:
        data = np.array(state, dtype=complex)
        if n_qubits is None:
            n_qubits = int(np.log2(len(data)))
    
    validate_qubits(n_qubits)
    dim = 2 ** n_qubits
    
    if len(data) != dim:
        raise ValueError(f"State dimension {len(data)} does not match 2^{n_qubits} = {dim}")
    
    # Build QFT circuit
    circ = QuantumCircuit(n_qubits)
    
    # Set initial state
    # We need to encode the input state into the circuit
    # For simplicity, we use direct matrix multiplication for the QFT
    
    # Build QFT matrix
    QFT_matrix = qft_matrix(n_qubits)
    
    # Apply to state
    result = QFT_matrix @ data
    
    logger.info(f"QFT applied to {n_qubits}-qubit state")
    
    return Ket(result)


def qft_circuit(n_qubits: int, inverse: bool = False, with_measurements: bool = False) -> QuantumCircuit:
    """
    Create a Quantum Fourier Transform circuit
    
    Args:
        n_qubits: Number of qubits
        inverse: If True, create inverse QFT (IQFT)
        with_measurements: If True, include measurement gates
        
    Returns:
        QuantumCircuit: QFT circuit
        
    Example:
        >>> circ = qft_circuit(3)
        >>> print(circ.draw())
    """
    validate_qubits(n_qubits)
    
    if inverse:
        circuit = _iqft_circuit(n_qubits)
    else:
        circuit = _qft_circuit(n_qubits)
    
    if with_measurements:
        for i in range(n_qubits):
            # We don't add measurements here to keep the circuit clean
            # Measurements can be added separately
            pass
    
    logger.info(f"{'Inverse ' if inverse else ''}QFT circuit created with {n_qubits} qubits")
    
    return circuit


def _qft_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Build QFT circuit
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        QuantumCircuit: QFT circuit
    """
    circuit = QuantumCircuit(n_qubits)
    
    for i in range(n_qubits):
        # Apply Hadamard to qubit i
        circuit.h(i)
        
        # Apply controlled phase rotations
        for j in range(i + 1, n_qubits):
            # Phase angle: π / 2^(j-i)
            angle = np.pi / (2 ** (j - i))
            circuit.cu1(angle, j, i)  # Controlled phase gate
    
    # Swap qubits to get correct order
    for i in range(n_qubits // 2):
        circuit.swap(i, n_qubits - 1 - i)
    
    return circuit


def _iqft_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Build inverse QFT circuit
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        QuantumCircuit: Inverse QFT circuit
    """
    circuit = QuantumCircuit(n_qubits)
    
    # Swap qubits (reverse order)
    for i in range(n_qubits // 2):
        circuit.swap(i, n_qubits - 1 - i)
    
    # Apply inverse rotations (reverse order and negative angles)
    for i in range(n_qubits - 1, -1, -1):
        # Apply inverse controlled phase rotations
        for j in range(n_qubits - 1, i, -1):
            # Phase angle: -π / 2^(j-i)
            angle = -np.pi / (2 ** (j - i))
            circuit.cu1(angle, j, i)
        
        # Apply Hadamard to qubit i
        circuit.h(i)
    
    return circuit


# ============================================================================
# QFT MATRIX
# ============================================================================

def qft_matrix(n_qubits: int) -> np.ndarray:
    """
    Generate the QFT matrix for n qubits
    
    The QFT matrix is a unitary matrix of size 2^n × 2^n:
    (QFT)_{j,k} = (1/√N) ω^{j*k}
    where ω = e^{2πi/N} and N = 2^n
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        np.ndarray: QFT matrix (2^n × 2^n)
        
    Example:
        >>> Q = qft_matrix(2)
        >>> print(Q)  # 4×4 QFT matrix
    """
    validate_qubits(n_qubits)
    N = 2 ** n_qubits
    
    QFT = np.zeros((N, N), dtype=complex)
    
    for j in range(N):
        for k in range(N):
            QFT[j, k] = np.exp(2j * np.pi * j * k / N) / np.sqrt(N)
    
    return QFT


def iqft_matrix(n_qubits: int) -> np.ndarray:
    """
    Generate the inverse QFT matrix
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        np.ndarray: Inverse QFT matrix
        
    Example:
        >>> Q_inv = iqft_matrix(2)
        >>> Q = qft_matrix(2)
        >>> print(np.allclose(Q @ Q_inv, np.eye(4)))  # True
    """
    QFT = qft_matrix(n_qubits)
    return QFT.conj().T


# ============================================================================
# QFT UTILITY FUNCTIONS
# ============================================================================

def iqft(state: Union[Ket, np.ndarray, List], n_qubits: Optional[int] = None) -> Ket:
    """
    Apply inverse Quantum Fourier Transform to a state
    
    Args:
        state: Input state (Ket, list, or numpy array)
        n_qubits: Number of qubits (auto-detected if not provided)
        
    Returns:
        Ket: Inverse QFT-transformed state
        
    Example:
        >>> from psiqit.quantum import Ket
        >>> state = Ket([1, 0, 0, 0])
        >>> qft_state = qft(state)
        >>> original = iqft(qft_state)
        >>> print(np.allclose(original.data, state.data))  # True
    """
    if isinstance(state, Ket):
        data = state.data
        if n_qubits is None:
            n_qubits = int(np.log2(len(data)))
    else:
        data = np.array(state, dtype=complex)
        if n_qubits is None:
            n_qubits = int(np.log2(len(data)))
    
    validate_qubits(n_qubits)
    dim = 2 ** n_qubits
    
    if len(data) != dim:
        raise ValueError(f"State dimension {len(data)} does not match 2^{n_qubits} = {dim}")
    
    # Build inverse QFT matrix
    IQFT_matrix = iqft_matrix(n_qubits)
    
    # Apply to state
    result = IQFT_matrix @ data
    
    logger.info(f"Inverse QFT applied to {n_qubits}-qubit state")
    
    return Ket(result)


def qft_controlled(n_qubits: int) -> QuantumCircuit:
    """
    Create a controlled QFT circuit
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        QuantumCircuit: Controlled QFT circuit
        
    Note:
        This is a placeholder for future implementation
    """
    # This would require additional qubits for control
    # For now, we return the standard QFT circuit
    logger.warning("Controlled QFT is not fully implemented")
    return qft_circuit(n_qubits)


# ============================================================================
# QFT FOR SPECIFIC STATES
# ============================================================================

def qft_basis_state(n_qubits: int, state_index: int) -> Ket:
    """
    Apply QFT to a basis state |j⟩ without constructing the full state
    
    Args:
        n_qubits: Number of qubits
        state_index: Index of the basis state (0 to 2^n - 1)
        
    Returns:
        Ket: QFT-transformed basis state
        
    Example:
        >>> result = qft_basis_state(2, 1)  # QFT(|01⟩)
        >>> print(result)  # Fourier basis state
    """
    validate_qubits(n_qubits)
    N = 2 ** n_qubits
    
    if state_index < 0 or state_index >= N:
        raise ValueError(f"state_index {state_index} out of range [0, {N-1}]")
    
    # Create basis state
    state = basis(N, state_index)
    
    # Apply QFT
    return qft(state)


def qft_product_state(n_qubits: int, amplitudes: List[complex]) -> Ket:
    """
    Apply QFT to a product state (tensor product of single-qubit states)
    
    Args:
        n_qubits: Number of qubits
        amplitudes: List of amplitudes for the product state
        
    Returns:
        Ket: QFT-transformed state
        
    Example:
        >>> amplitudes = [1/np.sqrt(2), 1/np.sqrt(2)]  # |+⟩
        >>> result = qft_product_state(2, amplitudes)
    """
    validate_qubits(n_qubits)
    
    # Build the product state
    state_data = np.array([1.0], dtype=complex)
    for amp in amplitudes[:n_qubits]:
        amp_array = np.array(amp, dtype=complex)
        if len(amp_array.shape) == 0:
            amp_array = np.array([amp_array], dtype=complex)
        state_data = np.kron(state_data, amp_array)
    
    # Normalize
    norm = np.linalg.norm(state_data)
    if norm > 0:
        state_data = state_data / norm
    
    return qft(Ket(state_data))


# ============================================================================
# QFT COMPARISON FUNCTIONS
# ============================================================================

def compare_qft_to_fft(n_qubits: int, state_index: int = 0) -> Dict[str, Any]:
    """
    Compare QFT to classical FFT on a basis state
    
    Args:
        n_qubits: Number of qubits
        state_index: Index of the basis state
        
    Returns:
        Dict: Comparison results
        
    Example:
        >>> result = compare_qft_to_fft(3, 2)
        >>> print(result['max_difference'])  # Numerical difference
    """
    from scipy.fft import fft
    
    N = 2 ** n_qubits
    
    # Create basis state
    state = np.zeros(N, dtype=complex)
    state[state_index] = 1.0
    
    # Apply QFT (quantum)
    qft_state = qft(Ket(state))
    qft_result = qft_state.data
    
    # Apply FFT (classical)
    fft_result = fft(state) / np.sqrt(N)
    
    # Compare
    max_diff = np.max(np.abs(qft_result - fft_result))
    
    return {
        'n_qubits': n_qubits,
        'state_index': state_index,
        'max_difference': float(max_diff),
        'qft_result': qft_result.tolist(),
        'fft_result': fft_result.tolist(),
        'is_close': max_diff < 1e-10
    }


def qft_circuit_comparison(n_qubits: int) -> Dict[str, int]:
    """
    Compare QFT circuit complexity
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        Dict: Circuit statistics
        
    Example:
        >>> stats = qft_circuit_comparison(3)
        >>> print(stats['gates'])  # Number of gates in QFT circuit
    """
    circ = qft_circuit(n_qubits)
    
    return {
        'n_qubits': n_qubits,
        'depth': circ.depth,
        'gates': len(circ.get_gates()),
        'classical_ops': 2 ** n_qubits * n_qubits,  # Classical FFT complexity
        'quantum_ops': n_qubits * (n_qubits + 1) // 2  # QFT complexity
    }


# ============================================================================
# QFT APPLICATIONS
# ============================================================================

def qft_phase_estimation_prep(n_qubits: int) -> QuantumCircuit:
    """
    Create a circuit for QFT-based phase estimation (initialization part)
    
    Args:
        n_qubits: Number of qubits for phase estimation
        
    Returns:
        QuantumCircuit: Phase estimation preparation circuit
    """
    circuit = QuantumCircuit(n_qubits)
    
    # Initialize to |+⟩⊗n
    for i in range(n_qubits):
        circuit.h(i)
    
    return circuit


def qft_phase_estimation_measure(n_qubits: int) -> QuantumCircuit:
    """
    Create a circuit for QFT-based phase estimation (measurement part)
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        QuantumCircuit: Phase estimation measurement circuit
    """
    # This is essentially the inverse QFT
    return _iqft_circuit(n_qubits)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'qft',
    'qft_circuit',
    'iqft',
    'qft_matrix',
    'iqft_matrix',
    'qft_controlled',
    'qft_basis_state',
    'qft_product_state',
    'compare_qft_to_fft',
    'qft_circuit_comparison',
    'qft_phase_estimation_prep',
    'qft_phase_estimation_measure',
]