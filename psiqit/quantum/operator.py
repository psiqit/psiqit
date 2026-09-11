#psiqit/quantum/operator.py

"""
Quantum Operators Module
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from ..math.qalgebra import Matrix, Vector, inner, eye, kron, commutator as q_commutator


# ============================================================================
# OPERATOR CLASS
# ============================================================================

class Operator:
    """
    Quantum operator/matrix representation
    
    This class represents quantum operators with support for
    unitary, Hermitian, and general operators.
    
    Example:
        >>> from psiqit.quantum import Operator, pauli_x
        >>> X = pauli_x()
        >>> print(X)
        Operator 'X' (dim=2)
        [[0.+0.j 1.+0.j]
         [1.+0.j 0.+0.j]]
    """
    
    def __init__(self, data: Union[List[List[complex]], np.ndarray], name: str = ""):
        """
        Initialize an operator
        
        Args:
            data: Matrix data (2D array)
            name: Name of the operator (optional)
        """
        self._data = np.array(data, dtype=complex)
        self._name = name
        
        # Check if square matrix
        if self._data.shape[0] != self._data.shape[1]:
            raise ValueError(f"Operator must be square, got shape {self._data.shape}")
        
        self._dim = self._data.shape[0]
        self._is_unitary = None
        self._is_hermitian = None
        
        # Lazy import logger برای جلوگیری از circular import
        from ..utils.logger import logger
        logger.debug(f"Operator '{name}' created with dim={self._dim}")
    
    @property
    def data(self) -> np.ndarray:
        """Get the matrix data"""
        return self._data
    
    @property
    def dim(self) -> int:
        """Get the dimension of the operator"""
        return self._dim
    
    @property
    def name(self) -> str:
        """Get the name of the operator"""
        return self._name
    
    @name.setter
    def name(self, value: str):
        """Set the name of the operator"""
        self._name = value
    
    @property
    def is_unitary(self) -> bool:
        """Check if operator is unitary (U†U = I)"""
        if self._is_unitary is None:
            self._is_unitary = self._validate_unitary()
        return self._is_unitary
    
    def _validate_unitary(self) -> bool:
        """Validate unitary without importing from utils"""
        identity = np.eye(self.dim, dtype=complex)
        product = self._data.conj().T @ self._data
        return np.allclose(product, identity, atol=1e-10)
    
    @property
    def is_hermitian(self) -> bool:
        """Check if operator is Hermitian (H† = H)"""
        if self._is_hermitian is None:
            self._is_hermitian = np.allclose(self._data, self._data.conj().T, atol=1e-10)
        return self._is_hermitian
    
    def dagger(self) -> 'Operator':
        """
        Return the conjugate transpose (Hermitian adjoint)
        
        Returns:
            Operator: Adjoint operator
        
        Example:
            >>> X = pauli_x()
            >>> Xd = X.dagger()
            >>> print(Xd == X)  # True (Pauli-X is Hermitian)
        """
        return Operator(self._data.conj().T, name=f"{self._name}†" if self._name else "")
    
    def trace(self) -> complex:
        """
        Return the trace of the operator
        
        Returns:
            complex: Trace value
        
        Example:
            >>> X = pauli_x()
            >>> print(X.trace())  # 0
        """
        return np.trace(self._data)
    
    def commutator(self, other: 'Operator') -> 'Operator':
        """
        Compute commutator [A, B] = A*B - B*A
        
        Args:
            other: Another operator
            
        Returns:
            Operator: Commutator
        
        Example:
            >>> X = pauli_x()
            >>> Z = pauli_z()
            >>> C = X.commutator(Z)  # -2iY
        """
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return Operator(
            self._data @ other._data - other._data @ self._data,
            name=f"[{self._name},{other._name}]" if self._name and other._name else ""
        )
    
    def anticommutator(self, other: 'Operator') -> 'Operator':
        """
        Compute anticommutator {A, B} = A*B + B*A
        
        Args:
            other: Another operator
            
        Returns:
            Operator: Anticommutator
        
        Example:
            >>> X = pauli_x()
            >>> Z = pauli_z()
            >>> A = X.anticommutator(Z)  # 0 (they anticommute)
        """
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return Operator(
            self._data @ other._data + other._data @ self._data,
            name=f"{{{self._name},{other._name}}}" if self._name and other._name else ""
        )
    
    def eigenvalues(self) -> List[complex]:
        """
        Compute eigenvalues of the operator
        
        Returns:
            List[complex]: Eigenvalues
        
        Example:
            >>> Z = pauli_z()
            >>> print(Z.eigenvalues())  # [1, -1]
        """
        return list(np.linalg.eigvals(self._data))
    
    def eigenvectors(self) -> Dict[str, Any]:
        """
        Compute eigenvectors and eigenvalues
        
        Returns:
            Dict: {'eigenvalues': [...], 'eigenvectors': [...]}
        
        Example:
            >>> Z = pauli_z()
            >>> result = Z.eigenvectors()
            >>> print(result['eigenvalues'])  # [1, -1]
        """
        eigvals, eigvecs = np.linalg.eig(self._data)
        return {
            'eigenvalues': list(eigvals),
            'eigenvectors': [eigvecs[:, i] for i in range(self.dim)]
        }
    
    def exp(self) -> 'Operator':
        """
        Compute matrix exponential
        
        Returns:
            Operator: Matrix exponential
        
        Example:
            >>> from psiqit.math import PI
            >>> X = pauli_x()
            >>> U = (1j * PI * X).exp()  # Rotation operator
        """
        return Operator(np.linalg.expm(self._data), name=f"exp({self._name})" if self._name else "")
    
    def show(self, precision: int = 3) -> None:
        """
        Pretty print the operator
        
        Args:
            precision: Number of decimal places
        
        Example:
            >>> X = pauli_x()
            >>> X.show()
            Operator 'X' (dim=2):
            [[0.+0.j 1.+0.j]
             [1.+0.j 0.+0.j]]
        """
        name_str = f" '{self._name}'" if self._name else ""
        print(f"Operator{name_str} (dim={self.dim}):")
        print(np.array2string(self._data, precision=precision, suppress_small=True))
    
    # ============================================================================
    # MAGIC METHODS
    # ============================================================================
    
    def __matmul__(self, other: Union['Operator', np.ndarray, Vector]) -> Union['Operator', np.ndarray]:
        """Matrix multiplication"""
        if isinstance(other, Operator):
            if self.dim != other.dim:
                raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
            return Operator(self._data @ other._data, name=f"{self._name}{other._name}" if self._name and other._name else "")
        elif isinstance(other, Vector):
            if self.dim != other.dim:
                raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
            return self._data @ other.data
        else:
            return self._data @ np.array(other)
    
    def __add__(self, other: 'Operator') -> 'Operator':
        """Operator addition"""
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return Operator(self._data + other._data, name=f"{self._name}+{other._name}" if self._name and other._name else "")
    
    def __sub__(self, other: 'Operator') -> 'Operator':
        """Operator subtraction"""
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return Operator(self._data - other._data, name=f"{self._name}-{other._name}" if self._name and other._name else "")
    
    def __mul__(self, scalar: Union[complex, float, int]) -> 'Operator':
        """Scalar multiplication"""
        return Operator(self._data * scalar, name=f"{scalar}{self._name}" if self._name else "")
    
    def __rmul__(self, scalar: Union[complex, float, int]) -> 'Operator':
        """Scalar multiplication (reverse)"""
        return self.__mul__(scalar)
    
    def __pow__(self, n: int) -> 'Operator':
        """Matrix power"""
        if n == 0:
            return Operator(np.eye(self.dim, dtype=complex), name=f"I")
        elif n == 1:
            return self
        elif n == 2:
            return Operator(self._data @ self._data, name=f"{self._name}²" if self._name else "")
        else:
            return Operator(np.linalg.matrix_power(self._data, n), name=f"{self._name}^{n}" if self._name else "")
    
    def __repr__(self) -> str:
        name_str = f" '{self._name}'" if self._name else ""
        return f"Operator{name_str} (dim={self.dim})"
    
    def __str__(self) -> str:
        name_str = f" '{self._name}'" if self._name else ""
        return f"Operator{name_str} (dim={self.dim}):\n{self._data}"
    
    def __eq__(self, other: 'Operator') -> bool:
        """Check equality of operators"""
        if not isinstance(other, Operator):
            return False
        return self.dim == other.dim and np.allclose(self._data, other._data)
    
    def __ne__(self, other: 'Operator') -> bool:
        """Check inequality of operators"""
        return not self.__eq__(other)
    
    def __neg__(self) -> 'Operator':
        """Negation of operator"""
        return Operator(-self._data, name=f"-{self._name}" if self._name else "")
    
    def __abs__(self) -> float:
        """Frobenius norm of operator"""
        return float(np.linalg.norm(self._data, 'fro'))
    
    def __len__(self) -> int:
        """Dimension of operator"""
        return self.dim
    
    def __getitem__(self, idx):
        """Get matrix element"""
        if isinstance(idx, tuple) and len(idx) == 2:
            return self._data[idx[0], idx[1]]
        elif isinstance(idx, int):
            return self._data[idx]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")
    def __setitem__(self, idx: Tuple[int, int], value: complex):
        """Set matrix element"""
        self._data[idx[0], idx[1]] = value
        # Reset cached properties
        self._is_unitary = None
        self._is_hermitian = None

# ============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# ============================================================================
# در انتهای فایل operator.py، قبل از __all__، اضافه کنید:

def trace(matrix):
    """
    Compute the trace of a matrix
    
    Args:
        matrix: Matrix (numpy array or Operator)
        
    Returns:
        complex: Trace value
        
    Example:
        >>> from psiqit.quantum import pauli_x, trace
        >>> X = pauli_x()
        >>> print(trace(X))  # 0
    """
    if isinstance(matrix, Operator):
        return matrix.trace()
    return np.trace(np.array(matrix, dtype=complex))
def dagger(matrix):
    """
    Return the conjugate transpose (Hermitian adjoint) of a matrix
    
    Args:
        matrix: Matrix (numpy array or Operator)
        
    Returns:
        numpy.ndarray or Operator: Conjugate transpose
        
    Example:
        >>> from psiqit.quantum import pauli_x, dagger
        >>> X = pauli_x()
        >>> Xd = dagger(X)
        >>> print(np.allclose(Xd.data, X.data))  # True (Pauli-X is Hermitian)
    """
    if isinstance(matrix, Operator):
        return matrix.dagger()
    return np.array(matrix, dtype=complex).conj().T

# ============================================================================
# PAULI MATRICES
# ============================================================================

def identity(dim: int = 2) -> Operator:
    """
    Identity operator
    
    Args:
        dim: Dimension of identity matrix
        
    Returns:
        Operator: Identity operator
        
    Example:
        >>> I = identity()
        >>> print(I.dim)  # 2
    """
    return Operator(np.eye(dim, dtype=complex), name="I")


def pauli_x() -> Operator:
    """
    Pauli-X (NOT) matrix
    
    X = [[0, 1], [1, 0]]
    
    Returns:
        Operator: Pauli-X operator
        
    Example:
        >>> X = pauli_x()
        >>> print(X.data)
        [[0 1]
         [1 0]]
    """
    return Operator([[0, 1], [1, 0]], name="X")


def pauli_y() -> Operator:
    """
    Pauli-Y matrix
    
    Y = [[0, -i], [i, 0]]
    
    Returns:
        Operator: Pauli-Y operator
    """
    return Operator([[0, -1j], [1j, 0]], name="Y")


def pauli_z() -> Operator:
    """
    Pauli-Z matrix
    
    Z = [[1, 0], [0, -1]]
    
    Returns:
        Operator: Pauli-Z operator
    """
    return Operator([[1, 0], [0, -1]], name="Z")


# ============================================================================
# SINGLE-QUBIT GATES
# ============================================================================

def hadamard() -> Operator:
    """
    Hadamard gate
    
    H = (1/√2)[[1, 1], [1, -1]]
    
    Returns:
        Operator: Hadamard gate
        
    Example:
        >>> H = hadamard()
        >>> print(H.is_unitary)  # True
    """
    h = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
    return Operator(h, name="H")


def phase(theta: float) -> Operator:
    """
    Phase gate
    
    R_φ = [[1, 0], [0, e^{iφ}]]
    
    Args:
        theta: Phase angle in radians
        
    Returns:
        Operator: Phase gate
        
    Example:
        >>> R = phase(np.pi/4)
        >>> print(R.name)  # Rφ(0.785)
    """
    return Operator([[1, 0], [0, np.exp(1j * theta)]], name=f"Rφ({theta:.3f})")


def s_gate() -> Operator:
    """
    S gate (phase gate with θ = π/2)
    
    S = [[1, 0], [0, i]]
    
    Returns:
        Operator: S gate
    """
    return Operator([[1, 0], [0, 1j]], name="S")


def t_gate() -> Operator:
    """
    T gate (phase gate with θ = π/4)
    
    T = [[1, 0], [0, e^{iπ/4}]]
    
    Returns:
        Operator: T gate
    """
    from ..math.qalgebra import PI
    return Operator([[1, 0], [0, np.exp(1j * PI / 4)]], name="T")


# ============================================================================
# ROTATION GATES
# ============================================================================

def rx(theta: float) -> Operator:
    """
    Rotation around X-axis
    
    R_x(θ) = e^{-iθX/2} = [[cos(θ/2), -i sin(θ/2)], [-i sin(θ/2), cos(θ/2)]]
    
    Args:
        theta: Rotation angle in radians
        
    Returns:
        Operator: X-rotation gate
        
    Example:
        >>> Rx = rx(np.pi/2)
        >>> print(Rx.is_unitary)  # True
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return Operator([[c, -1j * s], [-1j * s, c]], name=f"Rx({theta:.3f})")


def ry(theta: float) -> Operator:
    """
    Rotation around Y-axis
    
    R_y(θ) = e^{-iθY/2} = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
    
    Args:
        theta: Rotation angle in radians
        
    Returns:
        Operator: Y-rotation gate
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return Operator([[c, -s], [s, c]], name=f"Ry({theta:.3f})")


def rz(theta: float) -> Operator:
    """
    Rotation around Z-axis
    
    R_z(θ) = e^{-iθZ/2} = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]
    
    Args:
        theta: Rotation angle in radians
        
    Returns:
        Operator: Z-rotation gate
    """
    return Operator([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], name=f"Rz({theta:.3f})")


# ============================================================================
# TWO-QUBIT GATES
# ============================================================================

def cnot() -> Operator:
    """
    CNOT (Controlled-NOT) gate
    
    CNOT = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
    
    Returns:
        Operator: CNOT gate
        
    Example:
        >>> CNOT = cnot()
        >>> print(CNOT.is_unitary)  # True
    """
    return Operator([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], name="CNOT")


def cz() -> Operator:
    """
    CZ (Controlled-Z) gate
    
    CZ = diag(1, 1, 1, -1)
    
    Returns:
        Operator: CZ gate
    """
    return Operator(np.diag([1, 1, 1, -1]), name="CZ")


def swap() -> Operator:
    """
    SWAP gate
    
    SWAP = [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]
    
    Returns:
        Operator: SWAP gate
    """
    return Operator([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], name="SWAP")


# ============================================================================
# THREE-QUBIT GATES
# ============================================================================

def toffoli() -> Operator:
    """
    Toffoli (CCNOT) gate - Controlled-Controlled-NOT
    
    8x8 matrix that flips the target when both controls are 1
    
    Returns:
        Operator: Toffoli gate
        
    Example:
        >>> T = toffoli()
        >>> print(T.is_unitary)  # True
    """
    mat = np.eye(8, dtype=complex)
    # Flip last two bits: |110⟩ ↔ |111⟩
    mat[6, 6] = 0
    mat[7, 7] = 0
    mat[6, 7] = 1
    mat[7, 6] = 1
    return Operator(mat, name="TOFFOLI")


def fredkin() -> Operator:
    """
    Fredkin (CSWAP) gate - Controlled-SWAP
    
    8x8 matrix that swaps qubits 1 and 2 when control is 1
    
    Returns:
        Operator: Fredkin gate
    """
    mat = np.eye(8, dtype=complex)
    # Swap qubits 1 and 2 when control is 1
    mat[3, 3] = 0
    mat[5, 5] = 0
    mat[3, 5] = 1
    mat[5, 3] = 1
    return Operator(mat, name="FREDKIN")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def tensor_product(A: Operator, B: Operator) -> Operator:
    """
    Tensor product of two operators: A ⊗ B
    
    Args:
        A: First operator
        B: Second operator
        
    Returns:
        Operator: Tensor product
        
    Example:
        >>> X = pauli_x()
        >>> Z = pauli_z()
        >>> XZ = tensor_product(X, Z)
        >>> print(XZ.dim)  # 4
    """
    return Operator(np.kron(A.data, B.data), name=f"{A.name}⊗{B.name}" if A.name and B.name else "")


def expectation(operator: Operator, state: Vector) -> complex:
    """
    Compute expectation value ⟨ψ|O|ψ⟩
    
    Args:
        operator: Operator O
        state: Quantum state |ψ⟩
        
    Returns:
        complex: Expectation value
        
    Example:
        >>> from psiqit.quantum import Ket
        >>> Z = pauli_z()
        >>> state = Ket([1, 0])
        >>> print(expectation(Z, state))  # 1.0
    """
    if operator.dim != state.dim:
        raise ValueError(f"Dimension mismatch: operator {operator.dim} vs state {state.dim}")
    return np.vdot(state.data, operator.data @ state.data)


# ============================================================================
# ADDITIONAL USEFUL OPERATORS
# ============================================================================

def projector(state: Vector) -> Operator:
    """
    Create projection operator |ψ⟩⟨ψ|
    
    Args:
        state: Quantum state
        
    Returns:
        Operator: Projection operator
        
    Example:
        >>> from psiqit.quantum import Ket
        >>> state = Ket([1, 0])
        >>> P = projector(state)
        >>> print(P.data)
        [[1 0]
         [0 0]]
    """
    return Operator(np.outer(state.data, state.data.conj()), name=f"P_{state}")


def pauli_string(paulis: str) -> Operator:
    """
    Create a Pauli string operator (tensor product of Pauli matrices)
    
    Args:
        paulis: String of 'I', 'X', 'Y', 'Z' (e.g., 'XYZ')
        
    Returns:
        Operator: Pauli string operator
        
    Example:
        >>> P = pauli_string('XYZ')
        >>> print(P.dim)  # 8 (2^3)
    """
    pauli_map = {
        'I': identity(),
        'X': pauli_x(),
        'Y': pauli_y(),
        'Z': pauli_z()
    }
    
    result = None
    for p in paulis:
        if p not in pauli_map:
            raise ValueError(f"Invalid Pauli: {p}. Use I, X, Y, Z")
        op = pauli_map[p]
        if result is None:
            result = op
        else:
            result = tensor_product(result, op)
    
    return result


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Main class
    'Operator',
    
    # Pauli matrices
    'identity',
    'pauli_x',
    'pauli_y',
    'pauli_z',
    
    # Single-qubit gates
    'hadamard',
    'phase',
    's_gate',
    't_gate',
    
    # Rotation gates
    'rx',
    'ry',
    'rz',
    
    # Two-qubit gates
    'cnot',
    'cz',
    'swap',
    
    # Three-qubit gates
    'toffoli',
    'fredkin',
    
    # Utility functions
    'tensor_product',
    'expectation',
    'projector',
    'pauli_string',
    'dagger',
    'trace',
]