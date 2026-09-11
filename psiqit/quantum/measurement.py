 #psiqit/quantum/measurement.py

"""
Quantum Measurement Module
Measurement operations, POVM, projective measurement, and tomography
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from ..math.qalgebra import Matrix, Vector, inner, eye, dagger, trace, is_hermitian
from ..quantum.operator import Operator
from ..utils.logger import logger
from ..utils.validation import is_unitary as validate_unitary


# ============================================================================
# BASIC MEASUREMENT FUNCTIONS
# ============================================================================

def measure(
    state: Union[np.ndarray, Vector],
    shots: int = 1,
    basis: Optional[List[np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Measure a quantum state in a given basis
    
    Args:
        state: Quantum state (vector or Ket)
        shots: Number of measurements
        basis: Measurement basis (list of basis vectors)
               If None, uses computational basis
        
    Returns:
        Dict: Measurement results with counts and probabilities
        
    Example:
        >>> from psiqit.quantum import Ket
        >>> state = Ket([1/np.sqrt(2), 1/np.sqrt(2)])
        >>> result = measure(state, shots=1000)
        >>> print(result['counts'])
    """
    # Convert to numpy array if needed
    if isinstance(state, Vector):
        state = state.data
    elif hasattr(state, 'data'):
        state = state.data
    
    state = np.array(state, dtype=complex)
    
    # Normalize if needed
    norm = np.linalg.norm(state)
    if norm > 0 and abs(norm - 1.0) > 1e-10:
        state = state / norm
    
    # If no basis specified, use computational basis
    if basis is None:
        basis = [np.eye(len(state))[:, i] for i in range(len(state))]
    
    # Calculate probabilities
    probs = []
    for b in basis:
        b = np.array(b, dtype=complex)
        amp = np.vdot(b, state)
        probs.append(float(abs(amp) ** 2))
    
    # Normalize probabilities
    probs = np.array(probs)
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
    
    # Sample
    outcomes = np.random.choice(len(basis), size=shots, p=probs)
    
    # Count outcomes
    counts = {}
    for i, out in enumerate(outcomes):
        label = f"|{i}⟩" if basis is None else f"outcome_{i}"
        counts[label] = counts.get(label, 0) + 1
    
    logger.info(f"Measurement completed with {shots} shots")
    
    return {
        'counts': counts,
        'probabilities': probs.tolist(),
        'shots': shots,
        'dimension': len(state),
        'basis': basis
    }


def measure_observable(
    state: Union[np.ndarray, Vector],
    observable: Union[np.ndarray, Matrix],
    shots: int = 1
) -> Dict[str, Any]:
    """
    Measure an observable on a quantum state
    
    Args:
        state: Quantum state
        observable: Observable operator (Hermitian matrix)
        shots: Number of measurements
        
    Returns:
        Dict: Measurement results
        
    Example:
        >>> from psiqit.quantum import Ket, pauli_z
        >>> state = Ket([1, 0])
        >>> result = measure_observable(state, pauli_z(), shots=100)
        >>> print(result['expectation'])
    """
    # Convert to numpy if needed
    if isinstance(state, Vector):
        state = state.data
    elif hasattr(state, 'data'):
        state = state.data
    
    if isinstance(observable, Matrix):
        observable = observable.data
    elif isinstance(observable, Operator):
        observable = observable.data
    
    state = np.array(state, dtype=complex)
    observable = np.array(observable, dtype=complex)
    
    # Check dimensions
    if len(state) != observable.shape[0]:
        raise ValueError(f"Dimension mismatch: state {len(state)} vs observable {observable.shape[0]}")
    
    # Normalize state
    norm = np.linalg.norm(state)
    if norm > 0:
        state = state / norm
    
    # Compute eigenvalues and eigenvectors of observable
    eigvals, eigvecs = np.linalg.eigh(observable)
    
    # Project state onto eigenbasis
    probs = np.abs(np.vdot(eigvecs.T, state)) ** 2
    probs = probs / np.sum(probs)
    
    # Sample
    outcomes = np.random.choice(len(eigvals), size=shots, p=probs)
    
    # Count outcomes
    counts = {}
    for i, out in enumerate(outcomes):
        label = f"λ={eigvals[out]:.4f}"
        counts[label] = counts.get(label, 0) + 1
    
    # Calculate expectation
    expectation_value = np.vdot(state, observable @ state).real
    
    logger.info(f"Observable measurement completed with {shots} shots")
    
    return {
        'counts': counts,
        'probabilities': probs.tolist(),
        'eigenvalues': eigvals.tolist(),
        'shots': shots,
        'expectation': float(expectation_value),
        'outcomes': outcomes.tolist()
    }


def expectation(arg1, arg2) -> float:
    """
    Calculate expectation value ⟨ψ|O|ψ⟩
    
    Args:
        arg1: Observable operator OR Quantum state (Order-agnostic)
        arg2: Quantum state OR Observable operator (Order-agnostic)
        
    Returns:
        float: Expectation value
        
    Example:
        >>> from psiqit.quantum import Ket, pauli_z
        >>> state = Ket([1, 0])
        >>> exp = expectation(pauli_z(), state)  # Recommended order
        >>> exp2 = expectation(state, pauli_z()) # Also works! (Auto-corrected)
    """
    # Extract underlying numpy arrays safely
    data1 = arg1.data if hasattr(arg1, 'data') else np.array(arg1, dtype=complex)
    data2 = arg2.data if hasattr(arg2, 'data') else np.array(arg2, dtype=complex)
    
    # Smart dimension detection to prevent reshape errors and allow flexible argument order
    if data1.ndim == 1 and data2.ndim == 2:
        state_data = data1
        obs_data = data2
    elif data1.ndim == 2 and data2.ndim == 1:
        # User likely swapped the arguments: expectation(state, observable)
        logger.warning("Arguments swapped in expectation(): expected (observable, state), got (state, observable). Auto-correcting.")
        state_data = data2
        obs_data = data1
    else:
        raise ValueError(
            f"Invalid dimensions for expectation value: arg1 shape {data1.shape}, arg2 shape {data2.shape}. "
            "Expected one 1D state vector and one 2D observable matrix."
        )
    
    # Ensure state is a flat 1D array
    state_data = state_data.flatten()
    
    # Normalize state
    norm = np.linalg.norm(state_data)
    if norm > 0:
        state_data = state_data / norm
    
    # Calculate <psi|O|psi>
    result = np.vdot(state_data, obs_data @ state_data)
    return float(result.real)

def variance(
    observable: Union[np.ndarray, Matrix],
    state: Union[np.ndarray, Vector]
) -> float:
    """
    Calculate variance: ⟨O²⟩ - ⟨O⟩²
    
    Args:
        observable: Observable operator
        state: Quantum state
        
    Returns:
        float: Variance
        
    Example:
        >>> from psiqit.quantum import Ket, pauli_z
        >>> state = Ket([1/np.sqrt(2), 1/np.sqrt(2)])
        >>> var = variance(pauli_z(), state)
        >>> print(var)  # 1.0
    """
    exp_val = expectation(observable, state)
    exp2_val = expectation(observable @ observable, state)
    return float(exp2_val - exp_val ** 2)


def standard_deviation(
    observable: Union[np.ndarray, Matrix],
    state: Union[np.ndarray, Vector]
) -> float:
    """
    Calculate standard deviation: √variance
    
    Args:
        observable: Observable operator
        state: Quantum state
        
    Returns:
        float: Standard deviation
    """
    return np.sqrt(variance(observable, state))


# ============================================================================
# BORN RULE
# ============================================================================

def born_rule(amplitudes: np.ndarray) -> List[float]:
    """
    Apply Born rule: P(i) = |ψ_i|²
    
    Args:
        amplitudes: Complex amplitudes
        
    Returns:
        List[float]: Probabilities
        
    Example:
        >>> born_rule([1/np.sqrt(2), 1/np.sqrt(2)])  # [0.5, 0.5]
    """
    amplitudes = np.array(amplitudes, dtype=complex)
    probs = np.abs(amplitudes) ** 2
    
    # Normalize
    total = np.sum(probs)
    if total > 0:
        probs = probs / total
    
    return probs.tolist()


# ============================================================================
# POVM (Positive Operator-Valued Measure)
# ============================================================================

class POVM:
    """
    Positive Operator-Valued Measure (POVM)
    
    A POVM is a set of positive semidefinite operators {E_i} that sum to identity.
    
    Example:
        >>> # Create a POVM for Z-basis measurement
        >>> effects = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]
        >>> povm = POVM(effects, names=['|0⟩⟨0|', '|1⟩⟨1|'])
        >>> result = povm.measure(state, shots=1000)
    """
    
    def __init__(
        self,
        effects: List[np.ndarray],
        names: Optional[List[str]] = None
    ):
        """
        Initialize POVM
        
        Args:
            effects: List of POVM effect operators (positive semidefinite)
            names: Optional names for each outcome
        """
        self._effects = [np.array(e, dtype=complex) for e in effects]
        
        # Validate effects
        for i, E in enumerate(self._effects):
            if E.shape[0] != E.shape[1]:
                raise ValueError(f"Effect {i} must be square matrix")
        
        # Check completeness: Σ E_i = I
        dim = self._effects[0].shape[0]
        identity = np.eye(dim, dtype=complex)
        sum_effects = np.sum(self._effects, axis=0)
        
        if not np.allclose(sum_effects, identity):
            logger.warning("POVM effects do not sum to identity")
        
        self._dim = dim
        self._names = names or [f"E_{i}" for i in range(len(effects))]
        
        if len(self._names) != len(self._effects):
            raise ValueError("Number of names must match number of effects")
        
        logger.info(f"POVM initialized with {len(effects)} outcomes")
    
    @property
    def effects(self) -> List[np.ndarray]:
        """List of POVM effects"""
        return self._effects
    
    @property
    def names(self) -> List[str]:
        """Names of outcomes"""
        return self._names
    
    @property
    def n_outcomes(self) -> int:
        """Number of outcomes"""
        return len(self._effects)
    
    def measure(
        self,
        state: Union[np.ndarray, Vector],
        shots: int = 1
    ) -> Dict[str, Any]:
        """
        Perform POVM measurement on a state
        
        Args:
            state: Quantum state
            shots: Number of measurements
            
        Returns:
            Dict: Measurement results
        """
        # Convert to numpy if needed
        if isinstance(state, Vector):
            state = state.data
        elif hasattr(state, 'data'):
            state = state.data
        
        state = np.array(state, dtype=complex)
        
        # Normalize state
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
        
        # Calculate probabilities: P(i) = ⟨ψ|E_i|ψ⟩
        probs = []
        for E in self._effects:
            prob = np.vdot(state, E @ state).real
            probs.append(max(0, prob))  # Ensure non-negative
        
        # Normalize probabilities
        probs = np.array(probs)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        
        # Sample
        outcomes = np.random.choice(len(self._effects), size=shots, p=probs)
        
        # Count outcomes
        counts = {}
        for i, out in enumerate(outcomes):
            label = self._names[out] if out < len(self._names) else f"outcome_{out}"
            counts[label] = counts.get(label, 0) + 1
        
        logger.info(f"POVM measurement completed with {shots} shots")
        
        return {
            'counts': counts,
            'probabilities': probs.tolist(),
            'shots': shots,
            'dimension': self._dim,
            'names': self._names
        }


# ============================================================================
# PRE-DEFINED POVMS
# ============================================================================

def povm_z_basis() -> POVM:
    """
    POVM for Z-basis measurement (computational basis)
    
    Returns:
        POVM: Z-basis POVM
    
    Example:
        >>> povm = povm_z_basis()
        >>> result = povm.measure(state)
    """
    effects = [
        np.array([[1, 0], [0, 0]], dtype=complex),
        np.array([[0, 0], [0, 1]], dtype=complex)
    ]
    return POVM(effects, names=['|0⟩⟨0|', '|1⟩⟨1|'])


def povm_x_basis() -> POVM:
    """
    POVM for X-basis measurement
    
    Returns:
        POVM: X-basis POVM
    
    Example:
        >>> povm = povm_x_basis()
        >>> result = povm.measure(state)
    """
    # |+⟩⟨+| and |-⟩⟨-|
    effects = [
        0.5 * np.array([[1, 1], [1, 1]], dtype=complex),
        0.5 * np.array([[1, -1], [-1, 1]], dtype=complex)
    ]
    return POVM(effects, names=['|+⟩⟨+|', '|-⟩⟨-|'])


def povm_y_basis() -> POVM:
    """
    POVM for Y-basis measurement
    
    Returns:
        POVM: Y-basis POVM
    
    Example:
        >>> povm = povm_y_basis()
        >>> result = povm.measure(state)
    """
    # |i⟩⟨i| and |-i⟩⟨-i|
    effects = [
        0.5 * np.array([[1, -1j], [1j, 1]], dtype=complex),
        0.5 * np.array([[1, 1j], [-1j, 1]], dtype=complex)
    ]
    return POVM(effects, names=['|i⟩⟨i|', '|-i⟩⟨-i|'])


# ============================================================================
# PROJECTIVE MEASUREMENT
# ============================================================================

class ProjectiveMeasurement:
    """
    Projective measurement (von Neumann measurement)
    
    A projective measurement is defined by a set of orthogonal projectors
    that sum to identity.
    
    Example:
        >>> # Create projective measurement from Pauli-Z
        >>> from psiqit.quantum import pauli_z
        >>> pm = ProjectiveMeasurement.from_observable(pauli_z())
        >>> result = pm.measure(state, shots=1000)
    """
    
    def __init__(
        self,
        projectors: List[np.ndarray],
        eigenvalues: Optional[List[float]] = None
    ):
        """
        Initialize projective measurement
        
        Args:
            projectors: List of projection operators
            eigenvalues: Eigenvalues for each projector
        """
        self._projectors = [np.array(P, dtype=complex) for P in projectors]
        
        # Validate projectors
        for i, P in enumerate(self._projectors):
            if P.shape[0] != P.shape[1]:
                raise ValueError(f"Projector {i} must be square matrix")
            
            # Check if projector (P² = P)
            if not np.allclose(P @ P, P):
                logger.warning(f"Projector {i} is not idempotent")
        
        # Check completeness and orthogonality
        dim = self._projectors[0].shape[0]
        identity = np.eye(dim, dtype=complex)
        sum_projectors = np.sum(self._projectors, axis=0)
        
        if not np.allclose(sum_projectors, identity):
            logger.warning("Projectors do not sum to identity")
        
        # Check orthogonality
        for i in range(len(self._projectors)):
            for j in range(i + 1, len(self._projectors)):
                if not np.allclose(self._projectors[i] @ self._projectors[j], 
                                  np.zeros((dim, dim), dtype=complex)):
                    logger.warning(f"Projectors {i} and {j} are not orthogonal")
        
        self._dim = dim
        self._eigenvalues = eigenvalues or list(range(len(projectors)))
        
        if len(self._eigenvalues) != len(self._projectors):
            raise ValueError("Number of eigenvalues must match number of projectors")
        
        logger.info(f"Projective measurement initialized with {len(projectors)} outcomes")
    
    @classmethod
    def from_observable(
        cls,
        observable: Union[np.ndarray, Matrix, Operator]
    ) -> 'ProjectiveMeasurement':
        """
        Create projective measurement from an observable
        
        Args:
            observable: Hermitian observable
            
        Returns:
            ProjectiveMeasurement: Measurement in eigenbasis of observable
            
        Example:
            >>> from psiqit.quantum import pauli_z
            >>> pm = ProjectiveMeasurement.from_observable(pauli_z())
        """
        if isinstance(observable, Operator):
            observable = observable.data
        elif isinstance(observable, Matrix):
            observable = observable.data
        
        observable = np.array(observable, dtype=complex)
        
        # Compute spectral decomposition
        eigvals, eigvecs = np.linalg.eigh(observable)
        
        # Create projectors for each distinct eigenvalue
        projectors = []
        eigenvalues = []
        
        # Group degenerate eigenvalues
        unique_eigvals = np.unique(np.round(eigvals, 10))
        
        for val in unique_eigvals:
            mask = np.abs(eigvals - val) < 1e-10
            if np.sum(mask) > 0:
                # Projector onto eigenspace
                eigvecs_sub = eigvecs[:, mask]
                P = eigvecs_sub @ eigvecs_sub.conj().T
                projectors.append(P)
                eigenvalues.append(float(val))
        
        return cls(projectors, eigenvalues)
    
    @property
    def projectors(self) -> List[np.ndarray]:
        """List of projection operators"""
        return self._projectors
    
    @property
    def eigenvalues(self) -> List[float]:
        """Eigenvalues for each projector"""
        return self._eigenvalues
    
    @property
    def n_outcomes(self) -> int:
        """Number of outcomes"""
        return len(self._projectors)
    
    def measure(
        self,
        state: Union[np.ndarray, Vector],
        shots: int = 1
    ) -> Dict[str, Any]:
        """
        Perform projective measurement
        
        Args:
            state: Quantum state
            shots: Number of measurements
            
        Returns:
            Dict: Measurement results
        """
        # Convert to numpy if needed
        if isinstance(state, Vector):
            state = state.data
        elif hasattr(state, 'data'):
            state = state.data
        
        state = np.array(state, dtype=complex)
        
        # Normalize state
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
        
        # Calculate probabilities: P(i) = ⟨ψ|P_i|ψ⟩
        probs = []
        for P in self._projectors:
            prob = np.vdot(state, P @ state).real
            probs.append(max(0, prob))
        
        # Normalize probabilities
        probs = np.array(probs)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        
        # Sample
        outcomes = np.random.choice(len(self._projectors), size=shots, p=probs)
        
        # Count outcomes
        counts = {}
        for i, out in enumerate(outcomes):
            label = f"λ={self._eigenvalues[out]:.4f}"
            counts[label] = counts.get(label, 0) + 1
        
        logger.info(f"Projective measurement completed with {shots} shots")
        
        return {
            'counts': counts,
            'probabilities': probs.tolist(),
            'shots': shots,
            'dimension': self._dim,
            'eigenvalues': self._eigenvalues,
            'outcomes': outcomes.tolist()
        }


# ============================================================================
# MEASUREMENT STATISTICS
# ============================================================================

def measurement_statistics(
    state: Union[np.ndarray, Vector],
    shots: int = 1000
) -> Dict[str, Any]:
    """
    Compute comprehensive measurement statistics
    
    Args:
        state: Quantum state
        shots: Number of measurements
        
    Returns:
        Dict: Statistics including entropy, probabilities, etc.
        
    Example:
        >>> from psiqit.quantum import Ket
        >>> state = Ket([1/np.sqrt(2), 1/np.sqrt(2)])
        >>> stats = measurement_statistics(state, shots=10000)
    """
    # Convert to numpy if needed
    if isinstance(state, Vector):
        state = state.data
    elif hasattr(state, 'data'):
        state = state.data
    
    state = np.array(state, dtype=complex)
    
    # Normalize
    norm = np.linalg.norm(state)
    if norm > 0:
        state = state / norm
    
    # Probabilities
    probs = np.abs(state) ** 2
    probs = probs / np.sum(probs)
    
    # Shannon entropy
    ent = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
    
    # Max probability
    max_prob = np.max(probs)
    
    # Purity (for pure state = 1)
    purity = np.sum(probs ** 2)
    
    # Sample
    outcomes = np.random.choice(len(state), size=shots, p=probs)
    
    # Count outcomes
    counts = {}
    for out in outcomes:
        label = f"|{out}⟩"
        counts[label] = counts.get(label, 0) + 1
    
    logger.info(f"Measurement statistics computed with {shots} shots")
    
    return {
        'counts': counts,
        'probabilities': probs.tolist(),
        'shots': shots,
        'dimension': len(state),
        'entropy': float(ent),
        'max_probability': float(max_prob),
        'purity': float(purity),
        'most_likely': int(np.argmax(probs))
    }


# ============================================================================
# QUANTUM STATE TOMOGRAPHY
# ============================================================================

def state_tomography(
    measurement_data: Dict[str, Any],
    n_qubits: int = 1
) -> List[List[complex]]:
    """
    Reconstruct density matrix from measurement data (quantum state tomography)
    
    This is a simplified tomography for single-qubit states.
    
    Args:
        measurement_data: Dictionary with measurement counts from different bases
        n_qubits: Number of qubits (currently only 1 supported)
        
    Returns:
        List[List[complex]]: Reconstructed density matrix
        
    Example:
        >>> # Measure in X, Y, Z bases
        >>> data = {
        ...     'X': {'+': 520, '-': 480},
        ...     'Y': {'+': 490, '-': 510},
        ...     'Z': {'0': 600, '1': 400}
        ... }
        >>> rho = state_tomography(data)
    """
    if n_qubits != 1:
        raise NotImplementedError("Tomography currently only supported for single qubit")
    
    # Initialize density matrix
    rho = np.zeros((2, 2), dtype=complex)
    
    # Extract counts for each basis
    def get_prob(data: Dict, key: str) -> float:
        total = sum(data.values())
        if total == 0:
            return 0.0
        return data.get(key, 0) / total
    
    # Z-basis: ⟨Z⟩
    if 'Z' in measurement_data:
        p0 = get_prob(measurement_data['Z'], '0')
        p1 = get_prob(measurement_data['Z'], '1')
        p0 = get_prob(measurement_data['Z'], '|0⟩') or p0
        p1 = get_prob(measurement_data['Z'], '|1⟩') or p1
        
        exp_z = p0 - p1
        
        # Contribution to density matrix: (I + ⟨Z⟩ Z)/2
        rho[0, 0] += (1 + exp_z) / 2
        rho[1, 1] += (1 - exp_z) / 2
    
    # X-basis: ⟨X⟩
    if 'X' in measurement_data:
        p_plus = get_prob(measurement_data['X'], '+')
        p_minus = get_prob(measurement_data['X'], '-')
        p_plus = get_prob(measurement_data['X'], '|+⟩') or p_plus
        p_minus = get_prob(measurement_data['X'], '|-⟩') or p_minus
        
        exp_x = p_plus - p_minus
        
        # Contribution to density matrix: ⟨X⟩ X/2
        rho[0, 1] += exp_x / 2
        rho[1, 0] += exp_x / 2
    
    # Y-basis: ⟨Y⟩
    if 'Y' in measurement_data:
        p_plus = get_prob(measurement_data['Y'], '+')
        p_minus = get_prob(measurement_data['Y'], '-')
        p_plus = get_prob(measurement_data['Y'], '|i⟩') or p_plus
        p_minus = get_prob(measurement_data['Y'], '|-i⟩') or p_minus
        
        exp_y = p_plus - p_minus
        
        # Contribution to density matrix: ⟨Y⟩ Y/2
        rho[0, 1] -= 1j * exp_y / 2
        rho[1, 0] += 1j * exp_y / 2
    
    # Ensure Hermitian and trace 1
    rho = (rho + rho.conj().T) / 2
    trace_val = np.trace(rho).real
    if trace_val > 0:
        rho = rho / trace_val
    
    logger.info(f"State tomography completed for {n_qubits} qubit(s)")
    
    return rho.tolist()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Basic measurement
    'measure',
    'measure_observable',
    'expectation',
    'variance',
    'standard_deviation',
    
    # Born rule
    'born_rule',
    
    # POVM
    'POVM',
    'povm_z_basis',
    'povm_x_basis',
    'povm_y_basis',
    
    # Projective measurement
    'ProjectiveMeasurement',
    
    # Statistics and tomography
    'measurement_statistics',
    'state_tomography',
]