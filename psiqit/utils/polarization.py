# psiqit/utils/polarization.py

"""
Polarization Module
Jones vectors, Jones matrices, and polarization analysis
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from ..math.qalgebra import PI, SQRT2, cis, from_polar, to_polar
from ..quantum.state import Ket
from ..quantum.operator import Operator
from ..utils.logger import logger


# ============================================================================
# JONES VECTORS (POLARIZATION STATES)
# ============================================================================

def horizontal() -> np.ndarray:
    """
    Horizontal polarization state |H⟩ = [1, 0]ᵀ
    
    Returns:
        np.ndarray: Jones vector
        
    Example:
        >>> H = horizontal()
        >>> print(H)  # [1, 0]
    """
    return np.array([1.0, 0.0], dtype=complex)


def vertical() -> np.ndarray:
    """
    Vertical polarization state |V⟩ = [0, 1]ᵀ
    
    Returns:
        np.ndarray: Jones vector
        
    Example:
        >>> V = vertical()
        >>> print(V)  # [0, 1]
    """
    return np.array([0.0, 1.0], dtype=complex)


def diagonal(angle: float = PI/4) -> np.ndarray:
    """
    Linear polarization at a given angle
    
    Args:
        angle: Angle from horizontal (default: 45°)
        
    Returns:
        np.ndarray: Jones vector
        
    Example:
        >>> D = diagonal(PI/4)  # 45° diagonal
        >>> print(D)  # [1/√2, 1/√2]
    """
    return np.array([np.cos(angle), np.sin(angle)], dtype=complex)


def anti_diagonal() -> np.ndarray:
    """
    Anti-diagonal polarization state |A⟩ = [1/√2, -1/√2]ᵀ
    
    Returns:
        np.ndarray: Jones vector
        
    Example:
        >>> A = anti_diagonal()
        >>> print(A)  # [1/√2, -1/√2]
    """
    return np.array([1/SQRT2, -1/SQRT2], dtype=complex)


def circular_right() -> np.ndarray:
    """
    Right circular polarization |R⟩ = [1/√2, i/√2]ᵀ
    
    Returns:
        np.ndarray: Jones vector
        
    Example:
        >>> R = circular_right()
        >>> print(R)  # [1/√2, i/√2]
    """
    return np.array([1/SQRT2, 1j/SQRT2], dtype=complex)


def circular_left() -> np.ndarray:
    """
    Left circular polarization |L⟩ = [1/√2, -i/√2]ᵀ
    
    Returns:
        np.ndarray: Jones vector
        
    Example:
        >>> L = circular_left()
        >>> print(L)  # [1/√2, -i/√2]
    """
    return np.array([1/SQRT2, -1j/SQRT2], dtype=complex)


def elliptical(a: float, b: float, delta: float) -> np.ndarray:
    """
    Elliptical polarization state
    
    Args:
        a: Amplitude along x-axis
        b: Amplitude along y-axis
        delta: Phase difference between x and y components
        
    Returns:
        np.ndarray: Jones vector
        
    Example:
        >>> E = elliptical(1, 0.5, PI/2)
        >>> print(E)  # [1, 0.5i]
    """
    return np.array([a, b * np.exp(1j * delta)], dtype=complex)


# ============================================================================
# JONES MATRICES (OPTICAL ELEMENTS)
# ============================================================================

def polarizer(angle: float) -> np.ndarray:
    """
    Linear polarizer at a given angle
    
    Args:
        angle: Angle of the polarizer from horizontal
        
    Returns:
        np.ndarray: Jones matrix
        
    Example:
        >>> P = polarizer(PI/4)  # 45° polarizer
        >>> print(P.shape)  # (2, 2)
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c**2, c*s], [c*s, s**2]], dtype=complex)


def waveplate(retardance: float, angle: float = 0.0) -> np.ndarray:
    """
    Waveplate (retarder) at a given angle
    
    Args:
        retardance: Phase retardance (δ)
        angle: Angle of the waveplate from horizontal
        
    Returns:
        np.ndarray: Jones matrix
        
    Example:
        >>> QWP = waveplate(PI/2, 0)  # Quarter waveplate at 0°
        >>> print(QWP)
    """
    c = np.cos(angle)
    s = np.sin(angle)
    r = retardance / 2
    
    # Waveplate matrix in rotated basis
    W = np.array([
        [np.exp(1j * r), 0],
        [0, np.exp(-1j * r)]
    ], dtype=complex)
    
    # Rotation matrix
    R = np.array([[c, -s], [s, c]], dtype=complex)
    
    # Rotated waveplate
    return R @ W @ R.conj().T


def quarter_waveplate(angle: float = 0.0) -> np.ndarray:
    """
    Quarter waveplate (retardance = π/2)
    
    Args:
        angle: Angle of the waveplate from horizontal
        
    Returns:
        np.ndarray: Jones matrix
        
    Example:
        >>> QWP = quarter_waveplate(PI/4)
    """
    return waveplate(PI/2, angle)


def half_waveplate(angle: float = 0.0) -> np.ndarray:
    """
    Half waveplate (retardance = π)
    
    Args:
        angle: Angle of the waveplate from horizontal
        
    Returns:
        np.ndarray: Jones matrix
        
    Example:
        >>> HWP = half_waveplate(PI/8)
    """
    return waveplate(PI, angle)


def full_waveplate(angle: float = 0.0) -> np.ndarray:
    """
    Full waveplate (retardance = 2π)
    
    Args:
        angle: Angle of the waveplate from horizontal
        
    Returns:
        np.ndarray: Jones matrix
        
    Example:
        >>> FWP = full_waveplate(PI/4)
    """
    return waveplate(2 * PI, angle)


def rotator(angle: float) -> np.ndarray:
    """
    Optical rotator (rotation of polarization)
    
    Args:
        angle: Rotation angle
        
    Returns:
        np.ndarray: Jones matrix
        
    Example:
        >>> R = rotator(PI/4)  # 45° rotation
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=complex)


def linear_retarder(delta: float, angle: float = 0.0) -> np.ndarray:
    """
    Linear retarder (generalized waveplate)
    
    Args:
        delta: Phase retardance
        angle: Angle of the retarder from horizontal
        
    Returns:
        np.ndarray: Jones matrix
        
    Example:
        >>> LR = linear_retarder(PI/2, PI/4)
    """
    return waveplate(delta, angle)


def circular_dichroism(theta: float) -> np.ndarray:
    """
    Circular dichroism (different absorption for L and R)
    
    Args:
        theta: Dichroism parameter
        
    Returns:
        np.ndarray: Jones matrix
        
    Example:
        >>> CD = circular_dichroism(0.1)
    """
    return np.array([
        [np.cosh(theta), np.sinh(theta)],
        [np.sinh(theta), np.cosh(theta)]
    ], dtype=complex)


# ============================================================================
# POLARIZATION ANALYSIS
# ============================================================================

def stokes_parameters(jones: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute Stokes parameters from a Jones vector
    
    S₀ = |E_x|² + |E_y|²
    S₁ = |E_x|² - |E_y|²
    S₂ = 2 Re(E_x* E_y)
    S₃ = 2 Im(E_x* E_y)
    
    Args:
        jones: Jones vector
        
    Returns:
        Tuple[float, float, float, float]: (S₀, S₁, S₂, S₃)
        
    Example:
        >>> H = horizontal()
        >>> S = stokes_parameters(H)
        >>> print(S)  # (1, 1, 0, 0)
    """
    jones = np.array(jones, dtype=complex)
    
    if len(jones) != 2:
        raise ValueError("Jones vector must be 2-dimensional")
    
    # Normalize
    norm = np.linalg.norm(jones)
    if norm > 0:
        jones = jones / norm
    
    Ex, Ey = jones[0], jones[1]
    
    S0 = float(np.abs(Ex)**2 + np.abs(Ey)**2)
    S1 = float(np.abs(Ex)**2 - np.abs(Ey)**2)
    S2 = float(2 * np.real(Ex * np.conj(Ey)))
    S3 = float(2 * np.imag(Ex * np.conj(Ey)))
    
    return (S0, S1, S2, S3)


def degree_of_polarization(jones: np.ndarray) -> float:
    """
    Compute the degree of polarization (DOP)
    
    DOP = √(S₁² + S₂² + S₃²) / S₀
    
    Args:
        jones: Jones vector
        
    Returns:
        float: Degree of polarization (0-1)
        
    Example:
        >>> H = horizontal()
        >>> DOP = degree_of_polarization(H)
        >>> print(DOP)  # 1.0
    """
    S0, S1, S2, S3 = stokes_parameters(jones)
    
    if S0 == 0:
        return 0.0
    
    dop = np.sqrt(S1**2 + S2**2 + S3**2) / S0
    return float(min(1.0, dop))


def polarization_angle(jones: np.ndarray) -> float:
    """
    Compute the polarization angle (for linear polarization)
    
    θ = 0.5 * arctan(S₂/S₁)
    
    Args:
        jones: Jones vector
        
    Returns:
        float: Polarization angle in radians
        
    Example:
        >>> D = diagonal(PI/4)
        >>> angle = polarization_angle(D)
        >>> print(angle)  # 0.785 (π/4)
    """
    S1, S2 = stokes_parameters(jones)[1:3]
    
    if S1 == 0 and S2 == 0:
        return 0.0
    
    return 0.5 * np.arctan2(S2, S1)


def ellipticity(jones: np.ndarray) -> float:
    """
    Compute the ellipticity of the polarization state
    
    ε = 0.5 * arcsin(S₃/S₀)
    
    Args:
        jones: Jones vector
        
    Returns:
        float: Ellipticity in radians
        
    Example:
        >>> R = circular_right()
        >>> ellip = ellipticity(R)
        >>> print(ellip)  # π/4
    """
    S0, S3 = stokes_parameters(jones)[0], stokes_parameters(jones)[3]
    
    if S0 == 0:
        return 0.0
    
    return 0.5 * np.arcsin(np.clip(S3 / S0, -1, 1))


# ============================================================================
# POINCARE SPHERE
# ============================================================================

def to_poincare(jones: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a Jones vector to Poincaré sphere coordinates
    
    Args:
        jones: Jones vector
        
    Returns:
        Tuple[float, float, float]: (x, y, z) coordinates on Poincaré sphere
        
    Example:
        >>> H = horizontal()
        >>> x, y, z = to_poincare(H)
        >>> print(x, y, z)  # 1, 0, 0
    """
    S0, S1, S2, S3 = stokes_parameters(jones)
    
    if S0 == 0:
        return (0.0, 0.0, 0.0)
    
    x = S1 / S0
    y = S2 / S0
    z = S3 / S0
    
    return (float(x), float(y), float(z))


def from_poincare(x: float, y: float, z: float) -> np.ndarray:
    """
    Convert Poincaré sphere coordinates to a Jones vector
    
    Args:
        x: x-coordinate
        y: y-coordinate
        z: z-coordinate
        
    Returns:
        np.ndarray: Jones vector
        
    Example:
        >>> jones = from_poincare(1, 0, 0)  # Horizontal polarization
        >>> print(jones)  # [1, 0]
    """
    # Normalize
    r = np.sqrt(x**2 + y**2 + z**2)
    if r > 1e-10:
        x, y, z = x/r, y/r, z/r
    
    # Convert to Jones vector
    # |ψ⟩ = cos(θ/2)|H⟩ + e^{iφ} sin(θ/2)|V⟩
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x) if (x**2 + y**2) > 1e-10 else 0
    
    a = np.cos(theta/2)
    b = np.exp(1j * phi) * np.sin(theta/2)
    
    return np.array([a, b], dtype=complex)


# ============================================================================
# OPERATIONS
# ============================================================================

def apply_jones(matrix: np.ndarray, jones: np.ndarray) -> np.ndarray:
    """
    Apply a Jones matrix to a Jones vector
    
    Args:
        matrix: Jones matrix (2x2)
        jones: Jones vector (2x1)
        
    Returns:
        np.ndarray: Transformed Jones vector
        
    Example:
        >>> H = horizontal()
        >>> QWP = quarter_waveplate(0)
        >>> result = apply_jones(QWP, H)
        >>> print(result)  # Circular polarization
    """
    matrix = np.array(matrix, dtype=complex)
    jones = np.array(jones, dtype=complex)
    
    if matrix.shape != (2, 2):
        raise ValueError("Matrix must be 2x2")
    
    if len(jones) != 2:
        raise ValueError("Jones vector must be 2-dimensional")
    
    return matrix @ jones


def cascade_jones(matrices: List[np.ndarray]) -> np.ndarray:
    """
    Cascade multiple Jones matrices
    
    Args:
        matrices: List of Jones matrices
        
    Returns:
        np.ndarray: Combined Jones matrix
        
    Example:
        >>> QWP = quarter_waveplate(PI/4)
        >>> HWP = half_waveplate(0)
        >>> combined = cascade_jones([QWP, HWP])
        >>> print(combined.shape)  # (2, 2)
    """
    if not matrices:
        return np.eye(2, dtype=complex)
    
    result = np.eye(2, dtype=complex)
    
    # Multiply matrices in order (last applied first)
    for mat in reversed(matrices):
        result = np.array(mat, dtype=complex) @ result
    
    return result


def is_unitary_jones(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if a Jones matrix is unitary
    
    Args:
        matrix: Jones matrix
        tol: Tolerance for numerical checks
        
    Returns:
        bool: True if unitary, False otherwise
        
    Example:
        >>> QWP = quarter_waveplate(0)
        >>> is_unitary_jones(QWP)  # True
    """
    matrix = np.array(matrix, dtype=complex)
    
    if matrix.shape != (2, 2):
        return False
    
    identity = np.eye(2, dtype=complex)
    product = matrix @ matrix.conj().T
    
    return np.allclose(product, identity, atol=tol)


# ============================================================================
# ADDITIONAL POLARIZATION UTILITIES
# ============================================================================

def jones_to_ket(jones: np.ndarray) -> Ket:
    """
    Convert a Jones vector to a Ket state
    
    Args:
        jones: Jones vector
        
    Returns:
        Ket: Ket state
        
    Example:
        >>> H = horizontal()
        >>> ket = jones_to_ket(H)
        >>> print(ket)  # |0⟩
    """
    return Ket(jones)


def ket_to_jones(state: Ket) -> np.ndarray:
    """
    Convert a Ket state to a Jones vector
    
    Args:
        state: Ket state
        
    Returns:
        np.ndarray: Jones vector
        
    Example:
        >>> from psiqit.quantum import zero
        >>> jones = ket_to_jones(zero())
        >>> print(jones)  # [1, 0]
    """
    return state.data


def jones_to_operator(jones_matrix: np.ndarray, name: str = "") -> Operator:
    """
    Convert a Jones matrix to an Operator
    
    Args:
        jones_matrix: Jones matrix
        name: Name of the operator
        
    Returns:
        Operator: Operator object
    """
    return Operator(jones_matrix, name=name)


def operator_to_jones(operator: Operator) -> np.ndarray:
    """
    Convert an Operator to a Jones matrix
    
    Args:
        operator: Operator object
        
    Returns:
        np.ndarray: Jones matrix
    """
    return operator.data


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Jones vectors
    'horizontal',
    'vertical',
    'diagonal',
    'anti_diagonal',
    'circular_right',
    'circular_left',
    'elliptical',
    
    # Jones matrices
    'polarizer',
    'waveplate',
    'quarter_waveplate',
    'half_waveplate',
    'full_waveplate',
    'rotator',
    'linear_retarder',
    'circular_dichroism',
    
    # Polarization analysis
    'stokes_parameters',
    'degree_of_polarization',
    'polarization_angle',
    'ellipticity',
    
    # Poincare sphere
    'to_poincare',
    'from_poincare',
    
    # Operations
    'apply_jones',
    'cascade_jones',
    'is_unitary_jones',
    
    # Additional utilities
    'jones_to_ket',
    'ket_to_jones',
    'jones_to_operator',
    'operator_to_jones',
]