#psiqit/math/qalgebra.py

"""
Quantum Algebra Module
Linear algebra and quantum mechanics operations
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass

# ============================================================================
# CONSTANTS
# ============================================================================

PI = np.pi
TAU = 2 * np.pi
EULER = np.e
INF = np.inf
SQRT2 = np.sqrt(2)
SQRT3 = np.sqrt(3)
PHI = (1 + np.sqrt(5)) / 2
SQRT_PI = np.sqrt(np.pi)
SQRT_2PI = np.sqrt(2 * np.pi)

# Physical constants (in SI units)
HBAR = 1.054571817e-34
H_PLANCK = 6.62607015e-34
C = 299792458.0
E_CHARGE = 1.602176634e-19
M_ELECTRON = 9.1093837015e-31
M_PROTON = 1.67262192369e-27
K_B = 1.380649e-23
EPSILON_0 = 8.8541878128e-12
MU_0 = 4 * np.pi * 1e-7
GRAVITY = 9.80665


# ============================================================================
# INTERNAL VALIDATION FUNCTIONS (بدون وابستگی به utils)
# ============================================================================

def _is_unitary_internal(matrix, tol=1e-10):
    """Internal check for unitary matrix"""
    matrix = np.array(matrix, dtype=complex)
    if matrix.shape[0] != matrix.shape[1]:
        return False
    identity = np.eye(matrix.shape[0], dtype=complex)
    product = matrix.conj().T @ matrix
    return np.allclose(product, identity, atol=tol)


def _is_hermitian_internal(matrix, tol=1e-10):
    """Internal check for Hermitian matrix"""
    matrix = np.array(matrix, dtype=complex)
    if matrix.shape[0] != matrix.shape[1]:
        return False
    return np.allclose(matrix, matrix.conj().T, atol=tol)


# ============================================================================
# TYPES
# ============================================================================

@dataclass
class Vector:
    """Vector representation for quantum states and linear algebra"""
    data: np.ndarray
    dim: int = 0
    is_complex: bool = False
    
    def __post_init__(self):
        if self.dim == 0:
            self.dim = len(self.data)
        self.data = np.array(self.data, dtype=complex if self.is_complex else float)
        self.is_complex = np.iscomplexobj(self.data)
    
    def __add__(self, other: 'Vector') -> 'Vector':
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return Vector(self.data + other.data)
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return Vector(self.data - other.data)
    
    def __mul__(self, scalar: complex) -> 'Vector':
        return Vector(self.data * scalar)
    
    def __rmul__(self, scalar: complex) -> 'Vector':
        return self.__mul__(scalar)
    
    def __repr__(self) -> str:
        return f"Vector(dim={self.dim}, data={self.data.tolist()})"
    
    def __str__(self) -> str:
        return str(self.data)
    
    def __len__(self) -> int:
        return self.dim
    
    def __getitem__(self, index: int) -> complex:
        return self.data[index]
    
    def copy(self) -> 'Vector':
        return Vector(self.data.copy(), dim=self.dim)


@dataclass
class Matrix:
    """Matrix representation for operators and quantum gates"""
    data: np.ndarray
    shape: Tuple[int, int] = (0, 0)
    is_square: bool = False
    is_complex: bool = False
    
    def __post_init__(self):
        if self.shape == (0, 0):
            self.shape = self.data.shape
        self.data = np.array(self.data, dtype=complex if self.is_complex else float)
        self.is_square = self.shape[0] == self.shape[1]
        self.is_complex = np.iscomplexobj(self.data)
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        return Matrix(self.data + other.data)
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        return Matrix(self.data - other.data)
    
    def __matmul__(self, other: Union['Matrix', Vector]) -> Union['Matrix', Vector]:
        if isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            return Matrix(self.data @ other.data)
        elif isinstance(other, Vector):
            if self.shape[1] != other.dim:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.dim}")
            return Vector(self.data @ other.data)
        else:
            raise TypeError(f"Cannot multiply Matrix with {type(other)}")
    
    def __mul__(self, scalar: complex) -> 'Matrix':
        return Matrix(self.data * scalar)
    
    def __rmul__(self, scalar: complex) -> 'Matrix':
        return self.__mul__(scalar)
    
    def __repr__(self) -> str:
        return f"Matrix(shape={self.shape}, is_square={self.is_square})"
    
    def __str__(self) -> str:
        return str(self.data)
    
    def __getitem__(self, idx: Tuple[int, int]) -> complex:
        return self.data[idx[0], idx[1]]
    
    def copy(self) -> 'Matrix':
        return Matrix(self.data.copy())


# ============================================================================
# COMPLEX NUMBER OPERATIONS
# ============================================================================

def is_real(z: complex, tol: float = 1e-10) -> bool:
    return abs(z.imag) < tol


def phase(z: complex) -> float:
    if abs(z) < 1e-15:
        return 0.0
    return np.angle(z)


def magnitude(z: complex) -> float:
    return abs(z)


def cis(theta: float) -> complex:
    return np.cos(theta) + 1j * np.sin(theta)


def from_polar(r: float, theta: float) -> complex:
    return r * cis(theta)


def to_polar(z: complex) -> Tuple[float, float]:
    return abs(z), np.angle(z)


# ============================================================================
# MATRIX OPERATIONS
# ============================================================================

def eye(dim: int) -> Matrix:
    return Matrix(np.eye(dim, dtype=complex))


def zeros(rows: int, cols: int) -> Matrix:
    return Matrix(np.zeros((rows, cols), dtype=complex))


def add(A: Matrix, B: Matrix) -> Matrix:
    return A + B


def mul(A: Matrix, B: Matrix) -> Matrix:
    return A @ B


def transpose(A: Matrix) -> Matrix:
    return Matrix(A.data.T)


def dagger(A: Matrix) -> Matrix:
    return Matrix(A.data.conj().T)


def trace(A: Matrix) -> complex:
    if not A.is_square:
        raise ValueError("Trace is only defined for square matrices")
    return np.trace(A.data)


def kron(A: Matrix, B: Matrix) -> Matrix:
    return Matrix(np.kron(A.data, B.data))


# ============================================================================
# VECTOR OPERATIONS
# ============================================================================

def inner(v: Vector, w: Vector) -> complex:
    if v.dim != w.dim:
        raise ValueError(f"Dimension mismatch: {v.dim} vs {w.dim}")
    return np.vdot(v.data, w.data)


def outer(v: Vector, w: Vector) -> Matrix:
    if v.dim != w.dim:
        raise ValueError(f"Dimension mismatch: {v.dim} vs {w.dim}")
    return Matrix(np.outer(v.data, np.conj(w.data)))


def norm(v: Vector, order: int = 2) -> float:
    return float(np.linalg.norm(v.data, order))


def normalize(v: Vector) -> Vector:
    n = norm(v)
    if n == 0:
        raise ValueError("Cannot normalize zero vector")
    return Vector(v.data / n)


# ============================================================================
# ADVANCED MATRIX FUNCTIONS
# ============================================================================

def determinant(A: Matrix) -> complex:
    if not A.is_square:
        raise ValueError("Determinant is only defined for square matrices")
    return np.linalg.det(A.data)


def inverse(A: Matrix) -> Matrix:
    if not A.is_square:
        raise ValueError("Inverse is only defined for square matrices")
    return Matrix(np.linalg.inv(A.data))


def eigenvalues(A: Matrix) -> List[complex]:
    if not A.is_square:
        raise ValueError("Eigenvalues are only defined for square matrices")
    return list(np.linalg.eigvals(A.data))


def eigenvectors(A: Matrix) -> Tuple[List[complex], List[Vector]]:
    if not A.is_square:
        raise ValueError("Eigenvectors are only defined for square matrices")
    eigvals, eigvecs = np.linalg.eig(A.data)
    return list(eigvals), [Vector(eigvecs[:, i]) for i in range(len(eigvals))]


def expm(A: Matrix) -> Matrix:
    return Matrix(np.linalg.expm(A.data))


def sqrt_matrix(A: Matrix) -> Matrix:
    return Matrix(np.linalg.sqrtm(A.data))


# ============================================================================
# VALIDATION FUNCTIONS (با استفاده از توابع داخلی)
# ============================================================================

def is_unitary(A: Matrix, tol: float = 1e-10) -> bool:
    if not A.is_square:
        return False
    return _is_unitary_internal(A.data, tol)


def is_hermitian(A: Matrix, tol: float = 1e-10) -> bool:
    if not A.is_square:
        return False
    return _is_hermitian_internal(A.data, tol)


def is_positive(A, tol: float = 1e-10) -> bool:
    """Check if matrix is positive semidefinite"""
    # Check if A is a Matrix object
    if hasattr(A, 'is_square'):
        if not A.is_square:
            return False
        data = A.data
    else:
        data = np.array(A, dtype=complex)
        if data.shape[0] != data.shape[1]:
            return False
    
    try:
        eigvals = np.linalg.eigvalsh(data)
        return np.all(eigvals >= -tol)
    except:
        try:
            eigvals = np.linalg.eigvals(data)
            return np.all(eigvals >= -tol)
        except:
            return False
# ============================================================================
# PROBABILITY AND STATISTICS
# ============================================================================

def is_distribution(probs: np.ndarray, tol: float = 1e-10) -> bool:
    if len(probs) == 0:
        return False
    return np.all(probs >= -tol) and abs(np.sum(probs) - 1) < tol


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    total = np.sum(probs)
    if total == 0:
        raise ValueError("Sum of probabilities is zero")
    return probs / total


def sample(probs: np.ndarray, n_samples: int = 1, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    if not is_distribution(probs):
        probs = normalize_probs(probs)
    return np.random.choice(len(probs), size=n_samples, p=probs)


def entropy(probs: np.ndarray, base: str = 'e') -> float:
    if not is_distribution(probs):
        probs = normalize_probs(probs)
    log_func = {'e': np.log, '2': np.log2, '10': np.log10}.get(base)
    if log_func is None:
        raise ValueError(f"Unknown base: {base}")
    probs = probs[probs > 0]
    return -np.sum(probs * log_func(probs))


def born_rule(amplitudes: np.ndarray) -> np.ndarray:
    probs = np.abs(amplitudes) ** 2
    return normalize_probs(probs)


def expectation(operator: Matrix, state: Vector) -> complex:
    if not operator.is_square:
        raise ValueError("Operator must be a square matrix")
    if operator.shape[0] != state.dim:
        raise ValueError(f"Dimension mismatch: {operator.shape[0]} vs {state.dim}")
    return inner(state, operator @ state)


def variance(operator: Matrix, state: Vector) -> float:
    exp = expectation(operator, state)
    exp2 = expectation(operator @ operator, state)
    return float(exp2 - abs(exp)**2)


# ============================================================================
# QUANTUM OPERATORS TOOLS
# ============================================================================

def commutator(A: Matrix, B: Matrix) -> Matrix:
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    return A @ B - B @ A


def anticommutator(A: Matrix, B: Matrix) -> Matrix:
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    return A @ B + B @ A


def dimension(A: Matrix) -> int:
    if not A.is_square:
        raise ValueError("Dimension is only defined for square matrices")
    return A.shape[0]


def hilbert_space(dim: int) -> Dict[str, Any]:
    basis = [Vector(np.eye(dim)[:, i]) for i in range(dim)]
    identity = eye(dim)
    return {
        'dim': dim,
        'basis': basis,
        'identity': identity,
        'is_finite': True
    }


def display(A: Matrix, precision: int = 4) -> str:
    return np.array2string(A.data, precision=precision, suppress_small=True)


# ============================================================================
# UNIT CONVERSION
# ============================================================================

def eV_to_J(eV: float) -> float:
    return eV * E_CHARGE


def J_to_eV(J: float) -> float:
    return J / E_CHARGE


def nm_to_m(nm: float) -> float:
    return nm * 1e-9


def m_to_nm(m: float) -> float:
    return m / 1e-9


def energy_to_frequency(energy: float, hbar: Optional[float] = None) -> float:
    if hbar is None:
        hbar = HBAR
    return energy / hbar


def frequency_to_energy(frequency: float, hbar: Optional[float] = None) -> float:
    if hbar is None:
        hbar = HBAR
    return frequency * hbar


def wavelength_to_frequency(wavelength: float) -> float:
    return C / wavelength


def frequency_to_wavelength(frequency: float) -> float:
    return C / frequency


def wavelength_to_energy(wavelength: float) -> float:
    return H_PLANCK * C / wavelength


def energy_to_wavelength(energy: float) -> float:
    return H_PLANCK * C / energy


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'Vector', 'Matrix',
    'PI', 'TAU', 'EULER', 'INF', 'SQRT2', 'SQRT3', 'PHI',
    'SQRT_PI', 'SQRT_2PI',
    'HBAR', 'H_PLANCK', 'C', 'E_CHARGE', 'M_ELECTRON',
    'M_PROTON', 'K_B', 'EPSILON_0', 'MU_0', 'GRAVITY',
    'is_real', 'phase', 'magnitude', 'cis', 'from_polar', 'to_polar',
    'eye', 'zeros', 'add', 'mul', 'transpose', 'dagger', 'trace', 'kron',
    'inner', 'outer', 'norm', 'normalize',
    'determinant', 'inverse', 'eigenvalues', 'eigenvectors', 'expm', 'sqrt_matrix',
    'is_unitary', 'is_hermitian', 'is_positive',
    'is_distribution', 'normalize_probs', 'sample',
    'entropy', 'born_rule', 'expectation', 'variance',
    'commutator', 'anticommutator', 'dimension', 'hilbert_space', 'display',
    'eV_to_J', 'J_to_eV', 'nm_to_m', 'm_to_nm',
    'energy_to_frequency', 'frequency_to_energy',
    'wavelength_to_frequency', 'frequency_to_wavelength',
    'wavelength_to_energy', 'energy_to_wavelength',
]