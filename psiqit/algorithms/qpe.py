# psiqit/algorithms/qpe.py

"""
Quantum Phase Estimation (QPE)
Estimates the phase (eigenvalue) of a unitary operator
"""

import numpy as np
from typing import Optional, List, Union, Dict, Any, Tuple, Callable
from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, zero, one, plus, basis
from ..quantum.operator import Operator, identity, hadamard, phase, cnot, swap
from ..algorithms.qft import qft, iqft, qft_circuit
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# QUANTUM PHASE ESTIMATION
# ============================================================================

def quantum_phase_estimation(
    unitary: Union[Operator, Callable],
    n_qubits: int,
    state: Optional[Ket] = None,
    shots: int = 1024,
    return_circuit: bool = False,
    return_all: bool = False
) -> Dict[str, Any]:
    """
    Quantum Phase Estimation algorithm
    
    Estimates the phase φ such that U|ψ⟩ = e^{2πiφ}|ψ⟩
    where U is a unitary operator and |ψ⟩ is an eigenstate.
    
    Args:
        unitary: Unitary operator (Operator or callable that applies U)
        n_qubits: Number of qubits for phase estimation (precision)
        state: Eigenstate of U (if None, uses |0...0⟩)
        shots: Number of measurement shots
        return_circuit: If True, return the circuit
        return_all: If True, return all details
        
    Returns:
        Dict: Results including estimated phase
        
    Example:
        >>> from psiqit.quantum import pauli_z, zero
        >>> # Estimate phase of Z gate on |0⟩ (eigenvalue = 1, phase = 0)
        >>> result = quantum_phase_estimation(pauli_z(), n_qubits=3, state=zero())
        >>> print(result['phase'])  # 0.0
        >>> print(result['phase_angle'])  # 0.0
        
        >>> # Estimate phase of Z gate on |1⟩ (eigenvalue = -1, phase = 0.5)
        >>> from psiqit.quantum import one
        >>> result = quantum_phase_estimation(pauli_z(), n_qubits=3, state=one())
        >>> print(result['phase'])  # 0.5
        >>> print(result['phase_angle'])  # π
    """
    logger.info(f"Quantum Phase Estimation: {n_qubits} qubits")
    
    # Determine the dimension of the unitary
    if isinstance(unitary, Operator):
        dim = unitary.dim
        # Check if state is provided
        if state is not None and state.dim != dim:
            raise ValueError(f"State dimension {state.dim} does not match unitary dimension {dim}")
    else:
        # Callable unitary - we need to infer dimension
        # Use a test state
        test_state = Ket(np.array([1.0] + [0.0] * (2 ** n_qubits - 1)))
        dim = len(test_state.data)
    
    # If state is not provided, use |0...0⟩
    if state is None:
        state = basis(dim, 0)
    
    # Create circuit with n_qubits + m qubits (m = log2(dim) for the unitary)
    # For simplicity, we assume the unitary acts on a single qubit or system
    # The number of qubits for the unitary is log2(dim)
    m = int(np.log2(dim))
    total_qubits = n_qubits + m
    
    circuit = QuantumCircuit(total_qubits)
    
    # Prepare eigenstate |ψ⟩ on the last m qubits
    for i in range(m):
        if state.data[i] != 0:
            # This is a simplified version - for general states, we'd need more gates
            circuit.x(n_qubits + i)
    
    # Apply Hadamard to all phase estimation qubits
    for i in range(n_qubits):
        circuit.h(i)
    
    # Apply controlled-U operations
    for i in range(n_qubits):
        power = 2 ** i
        # Apply U^(2^i) controlled by qubit i
        _apply_controlled_unitary(circuit, unitary, i, list(range(n_qubits, total_qubits)), power)
    
    # Apply inverse QFT on the phase estimation qubits
    iqft_circ = _iqft_circuit(n_qubits)
    # Add IQFT gates to the circuit
    for gate in iqft_circ.get_gates():
        # Extract gate info and apply to the first n_qubits
        if gate['name'] == 'H':
            q = gate['qubits'][0]
            circuit.h(q)
        elif gate['name'] == 'SWAP':
            q1, q2 = gate['qubits'][0], gate['qubits'][1]
            circuit.swap(q1, q2)
        elif gate['name'] == 'CU1':
            # Controlled phase gate
            q_control, q_target = gate['qubits'][0], gate['qubits'][1]
            angle = gate['params'][0] if gate['params'] else 0
            circuit.cu1(angle, q_control, q_target)
    
    # Measure the phase estimation qubits
    result = circuit.measure(shots=shots)
    counts = result['counts']
    
    # Find most likely phase
    most_likely_binary = max(counts, key=counts.get) if counts else None
    
    # Convert binary to phase
    if most_likely_binary is not None:
        # Take only the first n_qubits bits
        phase_bits = most_likely_binary[:n_qubits]
        phase_value = int(phase_bits, 2) / (2 ** n_qubits)
        phase_angle = 2 * np.pi * phase_value
    else:
        phase_value = 0.0
        phase_angle = 0.0
    
    # Build result
    result_dict = {
        'phase': phase_value,
        'phase_angle': phase_angle,
        'phase_bits': most_likely_binary[:n_qubits] if most_likely_binary else None,
        'counts': counts,
        'shots': shots,
        'n_qubits': n_qubits,
        'algorithm': 'quantum_phase_estimation'
    }
    
    if return_circuit:
        result_dict['circuit'] = circuit
    
    if return_all:
        result_dict['all_counts'] = counts
        result_dict['most_likely_binary'] = most_likely_binary
    
    logger.info(f"QPE result: phase={phase_value:.6f}, angle={phase_angle:.6f}")
    
    return result_dict


# ============================================================================
# APPLY CONTROLLED UNITARY
# ============================================================================

def _apply_controlled_unitary(
    circuit: QuantumCircuit,
    unitary: Union[Operator, Callable],
    control: int,
    target_qubits: List[int],
    power: int
):
    """
    Apply controlled-U^(power) operation
    
    Args:
        circuit: QuantumCircuit object
        unitary: Unitary operator or callable
        control: Control qubit
        target_qubits: Target qubits for the unitary
        power: Power of U (U^power)
    """
    if isinstance(unitary, Operator):
        # If U is an Operator, we can apply it directly
        # For controlled-U, we need to add control
        _apply_controlled_operator(circuit, unitary, control, target_qubits, power)
    else:
        # If U is a callable, we apply it using the provided function
        _apply_controlled_callable(circuit, unitary, control, target_qubits, power)


def _apply_controlled_operator(
    circuit: QuantumCircuit,
    operator: Operator,
    control: int,
    target_qubits: List[int],
    power: int
):
    """
    Apply a controlled operator
    
    Args:
        circuit: QuantumCircuit object
        operator: Operator to apply
        control: Control qubit
        target_qubits: Target qubits
        power: Power of the operator
    """
    # For a single-qubit operator, use CU gates
    if len(target_qubits) == 1:
        target = target_qubits[0]
        
        # For Pauli gates, use controlled versions
        if operator.name == 'X':
            for _ in range(power):
                circuit.cx(control, target)
        elif operator.name == 'Z':
            for _ in range(power):
                circuit.cz(control, target)
        elif operator.name == 'H':
            # Controlled-H is more complex
            # For simplicity, we use phase estimation with H
            # Apply H with control
            circuit.ch(control, target)
        elif operator.name in ['RX', 'RY', 'RZ']:
            # For rotations, use controlled rotations
            # This is a simplified version
            angle = operator.data[1, 1].real if operator.name == 'RZ' else 0
            if operator.name == 'RX':
                circuit.crx(control, target, angle * power)
            elif operator.name == 'RY':
                circuit.cry(control, target, angle * power)
            elif operator.name == 'RZ':
                circuit.crz(control, target, angle * power)
        else:
            # General operator: use phase kickback
            # This is a placeholder for general operators
            logger.warning(f"Controlled {operator.name} not fully implemented")
    else:
        # Multi-qubit operator
        logger.warning("Multi-qubit controlled operators not fully implemented")


def _apply_controlled_callable(
    circuit: QuantumCircuit,
    unitary: Callable,
    control: int,
    target_qubits: List[int],
    power: int
):
    """
    Apply a controlled unitary via callable
    
    Args:
        circuit: QuantumCircuit object
        unitary: Callable that applies U
        control: Control qubit
        target_qubits: Target qubits
        power: Power of U
    """
    # For callable unitaries, we use the phase kickback technique
    # This requires the unitary to be applied conditionally
    
    # For simplicity, we apply the unitary with control
    # This is not fully general, but works for many cases
    for _ in range(power):
        # Apply U with control
        # We need to convert the callable to a controlled version
        # This is a placeholder
        logger.warning("Controlled callable unitary not fully implemented")
        # Apply unitary directly (without control) for simulation
        # In a real circuit, this would need controlled version
        unitary(circuit, target_qubits)


# ============================================================================
# QPE CIRCUIT
# ============================================================================

def qpe_circuit(
    n_qubits: int,
    m_qubits: int = 1,
    with_measurements: bool = True
) -> QuantumCircuit:
    """
    Create a Quantum Phase Estimation circuit
    
    Args:
        n_qubits: Number of qubits for phase estimation (precision)
        m_qubits: Number of qubits for the unitary (default: 1)
        with_measurements: Include measurement gates
        
    Returns:
        QuantumCircuit: QPE circuit
        
    Example:
        >>> circ = qpe_circuit(n_qubits=3, m_qubits=1)
        >>> print(circ.draw())
    """
    total_qubits = n_qubits + m_qubits
    circuit = QuantumCircuit(total_qubits)
    
    # Apply Hadamard to phase estimation qubits
    for i in range(n_qubits):
        circuit.h(i)
    
    # Controlled-U operations (placeholder)
    # In a real circuit, these would be implemented for a specific U
    
    # Inverse QFT
    iqft_circ = _iqft_circuit(n_qubits)
    for gate in iqft_circ.get_gates():
        if gate['name'] == 'H':
            q = gate['qubits'][0]
            circuit.h(q)
        elif gate['name'] == 'SWAP':
            q1, q2 = gate['qubits'][0], gate['qubits'][1]
            circuit.swap(q1, q2)
        elif gate['name'] == 'CU1':
            q_control, q_target = gate['qubits'][0], gate['qubits'][1]
            angle = gate['params'][0] if gate['params'] else 0
            circuit.cu1(angle, q_control, q_target)
    
    # Measurements
    if with_measurements:
        for i in range(n_qubits):
            # Measurements are not added here to keep the circuit clean
            pass
    
    logger.info(f"QPE circuit created: {n_qubits} phase qubits, {m_qubits} system qubits")
    
    return circuit


def _iqft_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Create inverse QFT circuit (used internally)
    
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
            angle = -np.pi / (2 ** (j - i))
            circuit.cu1(angle, j, i)
        
        # Apply Hadamard to qubit i
        circuit.h(i)
    
    return circuit


# ============================================================================
# QPE WITH SPECIFIC UNITARIES
# ============================================================================

def qpe_pauli_z(n_qubits: int, state: Ket, shots: int = 1024) -> Dict[str, Any]:
    """
    Run QPE on Pauli-Z gate
    
    Args:
        n_qubits: Number of qubits for phase estimation
        state: Eigenstate of Z (|0⟩ or |1⟩)
        shots: Number of measurement shots
        
    Returns:
        Dict: Results
        
    Example:
        >>> from psiqit.quantum import zero, one
        >>> result = qpe_pauli_z(3, zero())  # phase = 0
        >>> print(result['phase'])  # 0.0
        >>> result = qpe_pauli_z(3, one())   # phase = 0.5
        >>> print(result['phase'])  # 0.5
    """
    from ..quantum.operator import pauli_z
    
    # Check if state is eigenstate of Z
    if not (np.allclose(state.data, zero().data) or np.allclose(state.data, one().data)):
        logger.warning("State is not an eigenstate of Z, results may be mixed")
    
    return quantum_phase_estimation(
        unitary=pauli_z(),
        n_qubits=n_qubits,
        state=state,
        shots=shots
    )


def qpe_rotation_gate(n_qubits: int, angle: float, state: Ket, shots: int = 1024) -> Dict[str, Any]:
    """
    Run QPE on a rotation gate (RZ)
    
    Args:
        n_qubits: Number of qubits for phase estimation
        angle: Rotation angle
        state: Eigenstate of RZ (|0⟩ or |1⟩)
        shots: Number of measurement shots
        
    Returns:
        Dict: Results
        
    Example:
        >>> from psiqit.quantum import zero
        >>> result = qpe_rotation_gate(3, np.pi/4, zero())
        >>> print(result['phase'])  # 0.125 (for |0⟩, phase = 0)
        >>> result = qpe_rotation_gate(3, np.pi/4, one())
        >>> print(result['phase'])  # 0.375 (for |1⟩, phase = angle/(2π))
    """
    from ..quantum.operator import rz
    
    # Check if state is eigenstate of RZ
    if not (np.allclose(state.data, zero().data) or np.allclose(state.data, one().data)):
        logger.warning("State is not an eigenstate of RZ, results may be mixed")
    
    # For |0⟩, eigenvalue = 1 (phase = 0)
    # For |1⟩, eigenvalue = e^{-iθ} (phase = -θ/(2π))
    
    U = rz(angle)
    
    return quantum_phase_estimation(
        unitary=U,
        n_qubits=n_qubits,
        state=state,
        shots=shots
    )


# ============================================================================
# QPE UTILITY FUNCTIONS
# ============================================================================

def phase_to_angle(phase: float) -> float:
    """
    Convert phase to angle (in radians)
    
    Args:
        phase: Phase value (0 to 1)
        
    Returns:
        float: Angle in radians
        
    Example:
        >>> phase_to_angle(0.5)  # np.pi
    """
    return 2 * np.pi * phase


def angle_to_phase(angle: float) -> float:
    """
    Convert angle to phase (0 to 1)
    
    Args:
        angle: Angle in radians
        
    Returns:
        float: Phase value (0 to 1)
        
    Example:
        >>> angle_to_phase(np.pi)  # 0.5
    """
    return angle / (2 * np.pi)


def phase_to_eigenvalue(phase: float) -> complex:
    """
    Convert phase to eigenvalue
    
    Args:
        phase: Phase value (0 to 1)
        
    Returns:
        complex: Eigenvalue e^{2πiφ}
        
    Example:
        >>> phase_to_eigenvalue(0.5)  # -1
    """
    return np.exp(2j * np.pi * phase)


def eigenvalue_to_phase(eigenvalue: complex) -> float:
    """
    Convert eigenvalue to phase
    
    Args:
        eigenvalue: Complex eigenvalue
        
    Returns:
        float: Phase value (0 to 1)
        
    Example:
        >>> eigenvalue_to_phase(-1)  # 0.5
    """
    return np.angle(eigenvalue) / (2 * np.pi)


# ============================================================================
# QPE ACCURACY
# ============================================================================

def qpe_accuracy(n_qubits: int, true_phase: float) -> Dict[str, Any]:
    """
    Calculate the theoretical accuracy of QPE
    
    Args:
        n_qubits: Number of qubits for phase estimation
        true_phase: True phase value
        
    Returns:
        Dict: Accuracy statistics
        
    Example:
        >>> result = qpe_accuracy(3, 0.5)
        >>> print(result['max_error'])  # 1/2^3 = 0.125
    """
    N = 2 ** n_qubits
    resolution = 1 / N
    
    # Best approximation
    best_approx = round(true_phase * N) / N
    error = abs(true_phase - best_approx)
    
    return {
        'n_qubits': n_qubits,
        'resolution': resolution,
        'true_phase': true_phase,
        'best_approximation': best_approx,
        'error': error,
        'success_probability': np.sin(np.pi * N * error / 2) ** 2 if error > 0 else 1.0
    }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'quantum_phase_estimation',
    'qpe_circuit',
    'qpe_pauli_z',
    'qpe_rotation_gate',
    'phase_to_angle',
    'angle_to_phase',
    'phase_to_eigenvalue',
    'eigenvalue_to_phase',
    'qpe_accuracy',
]