# psiqit/utils/conversion.py

"""
Conversion Utilities Module
Convert between different quantum representations
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from ..quantum.state import Ket, Bra
from ..quantum.operator import Operator
from ..math.qalgebra import Matrix, Vector
from ..utils.logger import logger


# ============================================================================
# CONVERSION RESULT CLASS
# ============================================================================

@dataclass
class ConversionResult:
    """
    Result container for conversion operations
    
    Attributes:
        success: Whether conversion was successful
        result: Converted result
        message: Additional message
        details: Additional details
    """
    success: bool = True
    result: Any = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"ConversionResult(status={status}, message={self.message})"
    
    def __str__(self) -> str:
        lines = [
            "Conversion Result:",
            f"  Success: {self.success}",
            f"  Message: {self.message}",
        ]
        if self.details:
            lines.append(f"  Details: {self.details}")
        return "\n".join(lines)


# ============================================================================
# KET ↔ DENSITY MATRIX CONVERSIONS
# ============================================================================

def ket_to_density(state: Union[Ket, np.ndarray, List]) -> List[List[complex]]:
    """
    Convert a ket state to a density matrix
    
    ρ = |ψ⟩⟨ψ|
    
    Args:
        state: Quantum state (Ket, list, or numpy array)
        
    Returns:
        List[List[complex]]: Density matrix
        
    Example:
        >>> from psiqit.quantum import zero
        >>> rho = ket_to_density(zero())
        >>> print(rho)  # [[1, 0], [0, 0]]
    """
    if isinstance(state, Ket):
        data = state.data
    else:
        data = np.array(state, dtype=complex)
    
    # Normalize if needed
    norm = np.linalg.norm(data)
    if norm > 0:
        data = data / norm
    
    rho = np.outer(data, data.conj())
    return rho.tolist()


def density_to_ket(rho: Union[np.ndarray, List[List]], tol: float = 1e-10) -> ConversionResult:
    """
    Convert a density matrix to a ket state (if it's a pure state)
    
    Args:
        rho: Density matrix
        tol: Tolerance for numerical checks
        
    Returns:
        ConversionResult: Result with ket state if pure
        
    Example:
        >>> rho = [[1, 0], [0, 0]]
        >>> result = density_to_ket(rho)
        >>> print(result.result)  # |0⟩
    """
    rho = np.array(rho, dtype=complex)
    
    # Check if it's a valid density matrix
    if rho.shape[0] != rho.shape[1]:
        return ConversionResult(
            success=False,
            result=None,
            message="Matrix is not square"
        )
    
    # Check if it's Hermitian
    if not np.allclose(rho, rho.conj().T, atol=tol):
        return ConversionResult(
            success=False,
            result=None,
            message="Matrix is not Hermitian"
        )
    
    # Check if it's a pure state
    if not is_pure_state(rho, tol):
        return ConversionResult(
            success=False,
            result=None,
            message="State is not pure (mixed state)",
            details={'purity': _compute_purity(rho)}
        )
    
    # Get the dominant eigenvector
    eigvals, eigvecs = np.linalg.eigh(rho)
    idx = np.argmax(eigvals)
    state = Ket(eigvecs[:, idx])
    
    return ConversionResult(
        success=True,
        result=state,
        message="Successfully converted to ket state",
        details={
            'eigenvalue': float(eigvals[idx]),
            'purity': _compute_purity(rho)
        }
    )


def is_pure_state(rho: Union[np.ndarray, List[List]], tol: float = 1e-10) -> bool:
    """
    Check if a density matrix represents a pure state
    
    A pure state has Tr(ρ²) = 1
    
    Args:
        rho: Density matrix
        tol: Tolerance for numerical checks
        
    Returns:
        bool: True if pure, False otherwise
        
    Example:
        >>> rho = [[1, 0], [0, 0]]
        >>> is_pure_state(rho)  # True
        >>> rho = [[0.5, 0], [0, 0.5]]
        >>> is_pure_state(rho)  # False
    """
    rho = np.array(rho, dtype=complex)
    purity = _compute_purity(rho)
    return abs(purity - 1.0) < tol


def _compute_purity(rho: np.ndarray) -> float:
    """
    Compute purity: Tr(ρ²)
    
    Args:
        rho: Density matrix
        
    Returns:
        float: Purity (0-1)
    """
    return float(np.trace(rho @ rho).real)


# ============================================================================
# BASIS CONVERSIONS
# ============================================================================

def change_basis(state: Union[Ket, np.ndarray, List], basis: np.ndarray) -> List[complex]:
    """
    Change the basis of a state vector
    
    Args:
        state: State vector
        basis: New basis matrix (rows are basis vectors)
        
    Returns:
        List[complex]: State in new basis
        
    Example:
        >>> from psiqit.quantum import zero
        >>> from psiqit.quantum.operator import hadamard
        >>> H = hadamard().data
        >>> state = zero()
        >>> new_state = change_basis(state, H)
    """
    if isinstance(state, Ket):
        data = state.data
    else:
        data = np.array(state, dtype=complex)
    
    # Project onto the new basis
    new_state = np.conj(basis) @ data
    return new_state.tolist()


def to_computational_basis(state: Union[Ket, np.ndarray, List]) -> List[complex]:
    """
    Convert a state to the computational basis
    
    Args:
        state: State vector
        
    Returns:
        List[complex]: State in computational basis
        
    Example:
        >>> from psiqit.quantum import plus
        >>> state = plus()
        >>> comp = to_computational_basis(state)
        >>> print(comp)  # [0.707, 0.707]
    """
    if isinstance(state, Ket):
        return state.data.tolist()
    else:
        return np.array(state, dtype=complex).tolist()


def to_pauli_basis(matrix: Union[np.ndarray, List[List], Operator]) -> Dict[str, float]:
    """
    Decompose a 2x2 matrix into Pauli basis
    
    M = a₀I + a₁X + a₂Y + a₃Z
    
    Args:
        matrix: 2x2 matrix
        
    Returns:
        Dict[str, float]: Pauli coefficients {'I': a0, 'X': a1, 'Y': a2, 'Z': a3}
        
    Example:
        >>> from psiqit.quantum import pauli_x
        >>> X = pauli_x()
        >>> coeffs = to_pauli_basis(X)
        >>> print(coeffs)  # {'I': 0.0, 'X': 1.0, 'Y': 0.0, 'Z': 0.0}
    """
    if isinstance(matrix, Operator):
        data = matrix.data
    else:
        data = np.array(matrix, dtype=complex)
    
    if data.shape != (2, 2):
        raise ValueError("Matrix must be 2x2")
    
    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Compute coefficients
    coeff_I = np.trace(data @ I).real / 2
    coeff_X = np.trace(data @ X).real / 2
    coeff_Y = np.trace(data @ Y).real / 2
    coeff_Z = np.trace(data @ Z).real / 2
    
    return {
        'I': float(coeff_I),
        'X': float(coeff_X),
        'Y': float(coeff_Y),
        'Z': float(coeff_Z)
    }


def from_pauli_basis(coeffs: Dict[str, float]) -> List[List[complex]]:
    """
    Construct a 2x2 matrix from Pauli basis coefficients
    
    M = a₀I + a₁X + a₂Y + a₃Z
    
    Args:
        coeffs: Pauli coefficients
        
    Returns:
        List[List[complex]]: 2x2 matrix
        
    Example:
        >>> coeffs = {'I': 0.0, 'X': 1.0, 'Y': 0.0, 'Z': 0.0}
        >>> M = from_pauli_basis(coeffs)
        >>> print(M)  # [[0, 1], [1, 0]]
    """
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    M = (coeffs.get('I', 0.0) * I +
         coeffs.get('X', 0.0) * X +
         coeffs.get('Y', 0.0) * Y +
         coeffs.get('Z', 0.0) * Z)
    
    return M.tolist()


# ============================================================================
# MATRIX CONVERSIONS
# ============================================================================

def to_list(matrix: Union[np.ndarray, List, Operator]) -> List:
    """
    Convert to Python list
    
    Args:
        matrix: Matrix or operator
        
    Returns:
        List: Python list representation
    """
    if isinstance(matrix, Operator):
        return matrix.data.tolist()
    elif isinstance(matrix, np.ndarray):
        return matrix.tolist()
    else:
        return matrix


def to_operator(matrix: Union[np.ndarray, List], name: str = "") -> Operator:
    """
    Convert to Operator
    
    Args:
        matrix: Matrix data
        name: Name of the operator
        
    Returns:
        Operator: Operator object
        
    Example:
        >>> M = [[1, 0], [0, -1]]
        >>> op = to_operator(M, name="Z")
        >>> print(op.name)  # Z
    """
    return Operator(matrix, name=name)


def to_numpy(matrix: Union[np.ndarray, List, Operator]) -> np.ndarray:
    """
    Convert to numpy array
    
    Args:
        matrix: Matrix or operator
        
    Returns:
        np.ndarray: Numpy array
    """
    if isinstance(matrix, Operator):
        return matrix.data.copy()
    else:
        return np.array(matrix, dtype=complex)


def to_ket(amplitudes: Union[np.ndarray, List]) -> Ket:
    """
    Convert amplitudes to Ket state
    
    Args:
        amplitudes: Amplitude vector
        
    Returns:
        Ket: Ket state
        
    Example:
        >>> amplitudes = [1/np.sqrt(2), 1/np.sqrt(2)]
        >>> state = to_ket(amplitudes)
        >>> print(state)  # |+⟩
    """
    return Ket(amplitudes)


def to_bra(amplitudes: Union[np.ndarray, List]) -> Bra:
    """
    Convert amplitudes to Bra state
    
    Args:
        amplitudes: Amplitude vector
        
    Returns:
        Bra: Bra state
        
    Example:
        >>> amplitudes = [1/np.sqrt(2), 1/np.sqrt(2)]
        >>> bra = to_bra(amplitudes)
        >>> print(bra)  # ⟨+|
    """
    return Bra(np.conj(amplitudes))


def to_matrix(data: Union[np.ndarray, List]) -> Matrix:
    """
    Convert to Matrix object
    
    Args:
        data: Matrix data
        
    Returns:
        Matrix: Matrix object
    """
    return Matrix(data)


# ============================================================================
# BLOCH SPHERE CONVERSIONS
# ============================================================================

def to_bloch_coordinates(state: Union[Ket, np.ndarray, List]) -> Tuple[float, float, float]:
    """
    Convert a single qubit state to Bloch sphere coordinates
    
    |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
    
    Bloch coordinates:
    x = sin(θ)cos(φ)
    y = sin(θ)sin(φ)
    z = cos(θ)
    
    Args:
        state: Single qubit state
        
    Returns:
        Tuple[float, float, float]: (x, y, z) coordinates
        
    Example:
        >>> from psiqit.quantum import zero
        >>> x, y, z = to_bloch_coordinates(zero())
        >>> print(x, y, z)  # 0.0, 0.0, 1.0
    """
    if isinstance(state, Ket):
        data = state.data
    else:
        data = np.array(state, dtype=complex)
    
    if len(data) != 2:
        raise ValueError("State must be a single qubit (dimension 2)")
    
    # Normalize
    norm = np.linalg.norm(data)
    if norm > 0:
        data = data / norm
    
    a, b = data[0], data[1]
    
    # Bloch coordinates
    x = 2 * np.real(b * np.conj(a))
    y = 2 * np.imag(b * np.conj(a))
    z = np.abs(a)**2 - np.abs(b)**2
    
    return (float(x), float(y), float(z))


def from_bloch_coordinates(x: float, y: float, z: float) -> Ket:
    """
    Create a single qubit state from Bloch sphere coordinates
    
    Args:
        x: x-coordinate
        y: y-coordinate
        z: z-coordinate
        
    Returns:
        Ket: Single qubit state
        
    Example:
        >>> state = from_bloch_coordinates(0, 0, 1)  # |0⟩
        >>> print(state)
    """
    # Normalize
    r = np.sqrt(x**2 + y**2 + z**2)
    if r > 1e-10:
        x, y, z = x/r, y/r, z/r
    
    # Convert to state vector
    # |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x) if (x**2 + y**2) > 1e-10 else 0
    
    a = np.cos(theta/2)
    b = np.exp(1j * phi) * np.sin(theta/2)
    
    return Ket([a, b])


# ============================================================================
# VECTOR ↔ MATRIX CONVERSIONS
# ============================================================================

def vector_to_matrix(vec: Union[np.ndarray, List]) -> np.ndarray:
    """
    Convert a vector to a matrix (outer product with itself)
    
    M = vec ⊗ vec†
    
    Args:
        vec: Vector
        
    Returns:
        np.ndarray: Matrix
        
    Example:
        >>> vec = [1, 0]
        >>> M = vector_to_matrix(vec)
        >>> print(M)  # [[1, 0], [0, 0]]
    """
    vec = np.array(vec, dtype=complex)
    return np.outer(vec, vec.conj())


def matrix_to_vector(mat: Union[np.ndarray, List, Operator]) -> np.ndarray:
    """
    Convert a matrix to a vector (first column)
    
    For a density matrix, this extracts the state vector
    if the matrix is a pure state.
    
    Args:
        mat: Matrix
        
    Returns:
        np.ndarray: Vector
        
    Example:
        >>> M = [[1, 0], [0, 0]]
        >>> vec = matrix_to_vector(M)
        >>> print(vec)  # [1, 0]
    """
    if isinstance(mat, Operator):
        data = mat.data
    else:
        data = np.array(mat, dtype=complex)
    
    # For a pure state, the first column should be the state
    # (assuming the state is |0⟩ with phase)
    return data[:, 0]


# ============================================================================
# ADDITIONAL CONVERSION UTILITIES
# ============================================================================

def to_ket_list(states: List[Union[Ket, np.ndarray, List]]) -> List[Ket]:
    """
    Convert a list of states to Ket objects
    
    Args:
        states: List of states
        
    Returns:
        List[Ket]: List of Ket states
    """
    result = []
    for state in states:
        if isinstance(state, Ket):
            result.append(state)
        else:
            result.append(Ket(state))
    return result


def to_density_list(states: List[Union[Ket, np.ndarray, List]]) -> List[np.ndarray]:
    """
    Convert a list of states to density matrices
    
    Args:
        states: List of states
        
    Returns:
        List[np.ndarray]: List of density matrices
    """
    result = []
    for state in states:
        if isinstance(state, Ket):
            rho = np.outer(state.data, state.data.conj())
        else:
            data = np.array(state, dtype=complex)
            rho = np.outer(data, data.conj())
        result.append(rho)
    return result


def to_operator_list(matrices: List[Union[np.ndarray, List, Operator]]) -> List[Operator]:
    """
    Convert a list of matrices to Operator objects
    
    Args:
        matrices: List of matrices
        
    Returns:
        List[Operator]: List of Operators
    """
    result = []
    for mat in matrices:
        if isinstance(mat, Operator):
            result.append(mat)
        else:
            result.append(Operator(mat))
    return result


def convert_complex_to_real(matrix: Union[np.ndarray, List]) -> np.ndarray:
    """
    Convert a complex matrix to a real matrix by separating real and imaginary parts
    
    Args:
        matrix: Complex matrix
        
    Returns:
        np.ndarray: Real matrix (2x larger)
        
    Example:
        >>> complex_mat = [[1+1j, 2-2j], [3+3j, 4-4j]]
        >>> real_mat = convert_complex_to_real(complex_mat)
        >>> print(real_mat.shape)  # (4, 4)
    """
    mat = np.array(matrix, dtype=complex)
    n = mat.shape[0]
    
    real_mat = np.zeros((2*n, 2*n), dtype=float)
    
    # Block structure: [[Re(M), -Im(M)], [Im(M), Re(M)]]
    real_mat[:n, :n] = mat.real
    real_mat[:n, n:] = -mat.imag
    real_mat[n:, :n] = mat.imag
    real_mat[n:, n:] = mat.real
    
    return real_mat


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ConversionResult',
    'ket_to_density',
    'density_to_ket',
    'is_pure_state',
    'change_basis',
    'to_computational_basis',
    'to_pauli_basis',
    'from_pauli_basis',
    'to_list',
    'to_operator',
    'to_numpy',
    'to_ket',
    'to_bra',
    'to_matrix',
    'to_bloch_coordinates',
    'from_bloch_coordinates',
    'vector_to_matrix',
    'matrix_to_vector',
    'to_ket_list',
    'to_density_list',
    'to_operator_list',
    'convert_complex_to_real',
]