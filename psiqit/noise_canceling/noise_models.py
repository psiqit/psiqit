# psiqit/noise_canceling/noise_models.py

"""
Noise Models Module
Quantum noise channels and error mitigation techniques
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from ..quantum.state import Ket
from ..quantum.operator import Operator, identity, pauli_x, pauli_y, pauli_z, dagger, trace
from ..utils.logger import logger
from ..utils.validation import is_hermitian, is_unitary


# ============================================================================
# NOISE RESULT CLASS
# ============================================================================

@dataclass
class NoiseResult:
    """
    Result container for noise channel application
    
    Attributes:
        state: Resulting state (density matrix or Ket)
        fidelity: Fidelity with original state
        purity: Purity of the resulting state
        entropy: von Neumann entropy
        noise_amplitude: Noise parameter used
        success_rate: Success rate of the operation
    """
    state: Union[np.ndarray, Ket]
    fidelity: float = 1.0
    purity: float = 1.0
    entropy: float = 0.0
    noise_amplitude: float = 0.0
    success_rate: float = 1.0
    
    def __repr__(self) -> str:
        return f"NoiseResult(fidelity={self.fidelity:.4f}, purity={self.purity:.4f}, entropy={self.entropy:.4f})"
    
    def __str__(self) -> str:
        lines = [
            "Noise Channel Results:",
            f"  Fidelity: {self.fidelity:.6f}",
            f"  Purity: {self.purity:.6f}",
            f"  Entropy: {self.entropy:.6f}",
            f"  Noise Amplitude: {self.noise_amplitude:.6f}",
            f"  Success Rate: {self.success_rate:.2%}",
        ]
        return "\n".join(lines)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _to_density_matrix(state: Union[Ket, np.ndarray, Operator]) -> np.ndarray:
    """
    Convert a state to density matrix representation
    
    Args:
        state: Ket, Operator, or numpy array
        
    Returns:
        np.ndarray: Density matrix
    """
    if isinstance(state, Ket):
        return np.outer(state.data, state.data.conj())
    elif isinstance(state, Operator):
        return state.data
    else:
        # Assume it's already a density matrix
        return np.array(state, dtype=complex)


def _compute_fidelity(
    rho_ideal: np.ndarray,
    rho_noisy: np.ndarray
) -> float:
    """
    Compute fidelity between two density matrices
    
    F(ρ, σ) = [Tr(√(√ρ σ √ρ))]²
    
    Args:
        rho_ideal: Ideal density matrix
        rho_noisy: Noisy density matrix
        
    Returns:
        float: Fidelity (0-1)
    """
    # Use the simplified fidelity for pure states if possible
    # Check if rho_ideal is pure
    eigvals = np.linalg.eigvalsh(rho_ideal)
    if np.sum(eigvals > 1e-10) <= 1:
        # rho_ideal is pure, get the state vector
        eigvals, eigvecs = np.linalg.eigh(rho_ideal)
        idx = np.argmax(eigvals)
        psi = eigvecs[:, idx]
        fidelity = np.vdot(psi, rho_noisy @ psi).real
        return float(max(0, min(1, fidelity)))
    
    # General case: Uhlmann fidelity
    try:
        from scipy.linalg import sqrtm
        sqrt_rho = sqrtm(rho_ideal)
        sqrt_rho_sigma_sqrt_rho = sqrt_rho @ rho_noisy @ sqrt_rho
        fidelity = np.trace(sqrtm(sqrt_rho_sigma_sqrt_rho)).real ** 2
        return float(max(0, min(1, fidelity)))
    except:
        # Fallback: simple overlap
        return float(np.trace(rho_ideal @ rho_noisy).real)


def _compute_purity(rho: np.ndarray) -> float:
    """
    Compute purity of a density matrix
    
    P(ρ) = Tr(ρ²)
    
    Args:
        rho: Density matrix
        
    Returns:
        float: Purity (0-1)
    """
    return float(np.trace(rho @ rho).real)


def _compute_entropy(rho: np.ndarray, base: str = 'e') -> float:
    """
    Compute von Neumann entropy
    
    S(ρ) = -Tr(ρ log ρ)
    
    Args:
        rho: Density matrix
        base: Logarithm base
        
    Returns:
        float: Entropy
    """
    from ..info.entropy import von_neumann_entropy
    return von_neumann_entropy(rho, base)


# ============================================================================
# PAULI NOISE CHANNELS
# ============================================================================

def bit_flip_channel(
    state: Union[Ket, np.ndarray, Operator],
    p: float
) -> NoiseResult:
    """
    Apply bit-flip noise channel
    
    ρ → (1-p)ρ + p XρX
    
    Args:
        state: Input state
        p: Bit-flip probability (0-1)
        
    Returns:
        NoiseResult: Result with noisy state
        
    Example:
        >>> from psiqit.quantum import zero
        >>> result = bit_flip_channel(zero(), p=0.1)
        >>> print(result.fidelity)
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")
    
    rho = _to_density_matrix(state)
    X = pauli_x().data
    
    # Apply bit-flip channel
    rho_noisy = (1 - p) * rho + p * X @ rho @ X
    
    # Compute metrics
    fidelity = _compute_fidelity(rho, rho_noisy)
    purity = _compute_purity(rho_noisy)
    entropy = _compute_entropy(rho_noisy)
    
    logger.debug(f"Bit-flip channel applied with p={p:.3f}, fidelity={fidelity:.4f}")
    
    return NoiseResult(
        state=rho_noisy,
        fidelity=fidelity,
        purity=purity,
        entropy=entropy,
        noise_amplitude=p,
        success_rate=1.0
    )


def phase_flip_channel(
    state: Union[Ket, np.ndarray, Operator],
    p: float
) -> NoiseResult:
    """
    Apply phase-flip noise channel
    
    ρ → (1-p)ρ + p ZρZ
    
    Args:
        state: Input state
        p: Phase-flip probability (0-1)
        
    Returns:
        NoiseResult: Result with noisy state
        
    Example:
        >>> from psiqit.quantum import plus
        >>> result = phase_flip_channel(plus(), p=0.1)
        >>> print(result.fidelity)
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")
    
    rho = _to_density_matrix(state)
    Z = pauli_z().data
    
    # Apply phase-flip channel
    rho_noisy = (1 - p) * rho + p * Z @ rho @ Z
    
    # Compute metrics
    fidelity = _compute_fidelity(rho, rho_noisy)
    purity = _compute_purity(rho_noisy)
    entropy = _compute_entropy(rho_noisy)
    
    logger.debug(f"Phase-flip channel applied with p={p:.3f}, fidelity={fidelity:.4f}")
    
    return NoiseResult(
        state=rho_noisy,
        fidelity=fidelity,
        purity=purity,
        entropy=entropy,
        noise_amplitude=p,
        success_rate=1.0
    )


def bit_phase_flip_channel(
    state: Union[Ket, np.ndarray, Operator],
    p: float
) -> NoiseResult:
    """
    Apply bit-phase-flip noise channel (depolarizing with X and Z)
    
    ρ → (1-p)ρ + p YρY
    
    Args:
        state: Input state
        p: Bit-phase-flip probability (0-1)
        
    Returns:
        NoiseResult: Result with noisy state
        
    Example:
        >>> from psiqit.quantum import zero
        >>> result = bit_phase_flip_channel(zero(), p=0.1)
        >>> print(result.fidelity)
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")
    
    rho = _to_density_matrix(state)
    Y = pauli_y().data
    
    # Apply bit-phase-flip channel
    rho_noisy = (1 - p) * rho + p * Y @ rho @ Y
    
    # Compute metrics
    fidelity = _compute_fidelity(rho, rho_noisy)
    purity = _compute_purity(rho_noisy)
    entropy = _compute_entropy(rho_noisy)
    
    logger.debug(f"Bit-phase-flip channel applied with p={p:.3f}, fidelity={fidelity:.4f}")
    
    return NoiseResult(
        state=rho_noisy,
        fidelity=fidelity,
        purity=purity,
        entropy=entropy,
        noise_amplitude=p,
        success_rate=1.0
    )


def depolarizing_channel(
    state: Union[Ket, np.ndarray, Operator],
    p: float
) -> NoiseResult:
    """
    Apply depolarizing noise channel
    
    ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
    
    Args:
        state: Input state
        p: Depolarizing probability (0-1)
        
    Returns:
        NoiseResult: Result with noisy state
        
    Example:
        >>> from psiqit.quantum import zero
        >>> result = depolarizing_channel(zero(), p=0.1)
        >>> print(result.fidelity)
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")
    
    rho = _to_density_matrix(state)
    X = pauli_x().data
    Y = pauli_y().data
    Z = pauli_z().data
    
    # Apply depolarizing channel
    rho_noisy = (1 - p) * rho + (p / 3) * (X @ rho @ X + Y @ rho @ Y + Z @ rho @ Z)
    
    # Compute metrics
    fidelity = _compute_fidelity(rho, rho_noisy)
    purity = _compute_purity(rho_noisy)
    entropy = _compute_entropy(rho_noisy)
    
    logger.debug(f"Depolarizing channel applied with p={p:.3f}, fidelity={fidelity:.4f}")
    
    return NoiseResult(
        state=rho_noisy,
        fidelity=fidelity,
        purity=purity,
        entropy=entropy,
        noise_amplitude=p,
        success_rate=1.0
    )


# ============================================================================
# AMPLITUDE DAMPING CHANNEL
# ============================================================================

def amplitude_damping_channel(
    state: Union[Ket, np.ndarray, Operator],
    gamma: float
) -> NoiseResult:
    """
    Apply amplitude damping noise channel (energy relaxation)
    
    ρ → K₀ρK₀† + K₁ρK₁†
    where K₀ = [[1, 0], [0, √(1-γ)]], K₁ = [[0, √γ], [0, 0]]
    
    Args:
        state: Input state
        gamma: Damping rate (0-1)
        
    Returns:
        NoiseResult: Result with noisy state
        
    Example:
        >>> from psiqit.quantum import one
        >>> result = amplitude_damping_channel(one(), gamma=0.5)
        >>> print(result.fidelity)
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"gamma must be between 0 and 1, got {gamma}")
    
    rho = _to_density_matrix(state)
    
    # Kraus operators
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    
    # Apply amplitude damping channel
    rho_noisy = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T
    
    # Compute metrics
    fidelity = _compute_fidelity(rho, rho_noisy)
    purity = _compute_purity(rho_noisy)
    entropy = _compute_entropy(rho_noisy)
    
    logger.debug(f"Amplitude damping channel applied with gamma={gamma:.3f}, fidelity={fidelity:.4f}")
    
    return NoiseResult(
        state=rho_noisy,
        fidelity=fidelity,
        purity=purity,
        entropy=entropy,
        noise_amplitude=gamma,
        success_rate=1.0
    )


def phase_damping_channel(
    state: Union[Ket, np.ndarray, Operator],
    gamma: float
) -> NoiseResult:
    """
    Apply phase damping noise channel (dephasing)
    
    ρ → K₀ρK₀† + K₁ρK₁†
    where K₀ = [[1, 0], [0, √(1-γ)]], K₁ = [[0, 0], [0, √γ]]
    
    Args:
        state: Input state
        gamma: Damping rate (0-1)
        
    Returns:
        NoiseResult: Result with noisy state
        
    Example:
        >>> from psiqit.quantum import plus
        >>> result = phase_damping_channel(plus(), gamma=0.5)
        >>> print(result.fidelity)
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"gamma must be between 0 and 1, got {gamma}")
    
    rho = _to_density_matrix(state)
    
    # Kraus operators
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=complex)
    
    # Apply phase damping channel
    rho_noisy = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T
    
    # Compute metrics
    fidelity = _compute_fidelity(rho, rho_noisy)
    purity = _compute_purity(rho_noisy)
    entropy = _compute_entropy(rho_noisy)
    
    logger.debug(f"Phase damping channel applied with gamma={gamma:.3f}, fidelity={fidelity:.4f}")
    
    return NoiseResult(
        state=rho_noisy,
        fidelity=fidelity,
        purity=purity,
        entropy=entropy,
        noise_amplitude=gamma,
        success_rate=1.0
    )


# ============================================================================
# READOUT ERROR
# ============================================================================

def readout_error(
    counts: Dict[str, int],
    error_matrix: np.ndarray,
    shots: int = None
) -> Dict[str, int]:
    """
    Apply readout error to measurement counts
    
    Args:
        counts: Original measurement counts
        error_matrix: Readout error matrix (n_states x n_states)
        shots: Total number of shots (if None, inferred from counts)
        
    Returns:
        Dict[str, int]: Corrected counts
        
    Example:
        >>> counts = {'00': 900, '11': 100}
        >>> error_matrix = np.array([[0.95, 0.05], [0.05, 0.95]])
        >>> corrected = readout_error(counts, error_matrix)
    """
    n_states = error_matrix.shape[0]
    
    # Convert counts to vector
    state_labels = sorted(counts.keys())
    count_vec = np.array([counts.get(label, 0) for label in state_labels])
    
    if shots is None:
        shots = np.sum(count_vec)
    
    # Apply error matrix (inverse)
    try:
        inv_error = np.linalg.inv(error_matrix)
        corrected_vec = inv_error @ count_vec
        corrected_vec = np.maximum(corrected_vec, 0)  # Remove negative values
    except:
        logger.warning("Error matrix is singular, using pseudoinverse")
        inv_error = np.linalg.pinv(error_matrix)
        corrected_vec = inv_error @ count_vec
        corrected_vec = np.maximum(corrected_vec, 0)
    
    # Normalize to preserve total shots
    if np.sum(corrected_vec) > 0:
        corrected_vec = corrected_vec / np.sum(corrected_vec) * shots
    
    # Convert back to dictionary
    corrected_counts = {}
    for label, count in zip(state_labels, corrected_vec):
        corrected_counts[label] = int(round(count))
    
    return corrected_counts


# ============================================================================
# ERROR MITIGATION TECHNIQUES
# ============================================================================

def zero_noise_extrapolation(
    results: List[NoiseResult],
    order: int = 1
) -> float:
    """
    Zero-noise extrapolation (ZNE)
    
    Extrapolate to zero noise using Richardson extrapolation.
    
    Args:
        results: List of NoiseResult at different noise levels
        order: Order of extrapolation
        
    Returns:
        float: Extrapolated value at zero noise
        
    Example:
        >>> results = []
        >>> for p in [0.1, 0.2, 0.3]:
        ...     result = depolarizing_channel(state, p)
        ...     results.append(result)
        >>> value = zero_noise_extrapolation(results)
    """
    if len(results) < order + 1:
        logger.warning("Not enough data points for extrapolation")
        return results[-1].fidelity
    
    # Extract noise amplitudes and fidelities
    x = np.array([r.noise_amplitude for r in results])
    y = np.array([r.fidelity for r in results])
    
    # Richardson extrapolation
    if order == 1:
        # Linear extrapolation
        coef = np.polyfit(x, y, 1)
        return float(np.polyval(coef, 0))
    else:
        # Higher order extrapolation
        coef = np.polyfit(x, y, order)
        return float(np.polyval(coef, 0))


def probabilistic_error_cancellation(
    noise_result: NoiseResult,
    error_rate: float,
    mitigation_strength: float = 1.0
) -> NoiseResult:
    """
    Probabilistic error cancellation (PEC)
    
    Args:
        noise_result: Result from a noise channel
        error_rate: Estimated error rate
        mitigation_strength: Strength of mitigation (0-1)
        
    Returns:
        NoiseResult: Mitigated result
        
    Example:
        >>> result = depolarizing_channel(state, p=0.1)
        >>> mitigated = probabilistic_error_cancellation(result, error_rate=0.1)
    """
    # This is a simplified implementation of PEC
    # In practice, this would involve sampling from a quasi-probability distribution
    
    # Simple mitigation: rescale the state
    rho = noise_result.state
    if isinstance(rho, np.ndarray):
        # Try to invert the noise effect
        # This is a placeholder - full PEC requires knowledge of the noise channel
        rho_mitigated = rho.copy()
        
        # Simple rescaling
        if mitigation_strength > 0:
            correction = 1 / (1 + error_rate * mitigation_strength)
            # Apply correction to off-diagonal elements
            for i in range(rho.shape[0]):
                for j in range(rho.shape[1]):
                    if i != j:
                        rho_mitigated[i, j] = rho[i, j] * correction
        
        # Ensure Hermitian and trace 1
        rho_mitigated = (rho_mitigated + rho_mitigated.conj().T) / 2
        trace_val = np.trace(rho_mitigated).real
        if trace_val > 0:
            rho_mitigated = rho_mitigated / trace_val
        
        return NoiseResult(
            state=rho_mitigated,
            fidelity=noise_result.fidelity * (1 + error_rate * mitigation_strength),
            purity=_compute_purity(rho_mitigated),
            entropy=_compute_entropy(rho_mitigated),
            noise_amplitude=noise_result.noise_amplitude * (1 - mitigation_strength),
            success_rate=1.0
        )
    
    return noise_result


# ============================================================================
# NOISE MODEL COMPOSITION
# ============================================================================

def compose_noise_channels(
    state: Union[Ket, np.ndarray, Operator],
    channels: List[Tuple[str, float]]
) -> NoiseResult:
    """
    Compose multiple noise channels
    
    Args:
        state: Input state
        channels: List of (channel_name, parameter) tuples
        
    Returns:
        NoiseResult: Result after applying all channels
        
    Example:
        >>> result = compose_noise_channels(
        ...     state,
        ...     [('bit_flip', 0.1), ('phase_flip', 0.05)]
        ... )
    """
    channel_map = {
        'bit_flip': bit_flip_channel,
        'phase_flip': phase_flip_channel,
        'bit_phase_flip': bit_phase_flip_channel,
        'depolarizing': depolarizing_channel,
        'amplitude_damping': amplitude_damping_channel,
        'phase_damping': phase_damping_channel,
    }
    
    current_state = state
    result = None
    
    for channel_name, param in channels:
        if channel_name not in channel_map:
            logger.warning(f"Unknown channel: {channel_name}, skipping")
            continue
        
        channel = channel_map[channel_name]
        result = channel(current_state, param)
        current_state = result.state
    
    if result is None:
        # No channels applied
        rho = _to_density_matrix(state)
        result = NoiseResult(state=rho)
    
    return result


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'NoiseResult',
    'bit_flip_channel',
    'phase_flip_channel',
    'bit_phase_flip_channel',
    'depolarizing_channel',
    'amplitude_damping_channel',
    'phase_damping_channel',
    'readout_error',
    'zero_noise_extrapolation',
    'probabilistic_error_cancellation',
    'compose_noise_channels',
] 