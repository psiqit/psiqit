# psiqit/info/entanglement.py

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from ..quantum.state import Ket
from ..quantum.operator import Operator
from ..math.qalgebra import Matrix, trace, is_positive, eigenvalues
from ..utils.logger import logger
from ..utils.validation import is_hermitian


# ============================================================================
# ENTANGLEMENT RESULT CLASS
# ============================================================================

@dataclass
class EntanglementResult:
    """
    Result container for entanglement measures
    
    Attributes:
        value: Numerical value of the entanglement measure
        measure: Name of the measure used
        is_entangled: Whether the state is entangled
        details: Additional information
        dims: Subsystem dimensions
    """
    value: float = 0.0
    measure: str = ""
    is_entangled: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    dims: Optional[List[int]] = None
    
    def __repr__(self) -> str:
        return f"EntanglementResult(value={self.value:.6f}, measure={self.measure}, is_entangled={self.is_entangled})"
    
    def __str__(self) -> str:
        lines = [
            "Entanglement Results:",
            f"  Measure: {self.measure}",
            f"  Value: {self.value:.6f}",
            f"  Is Entangled: {self.is_entangled}",
        ]
        if self.dims:
            lines.append(f"  Dimensions: {self.dims}")
        if self.details:
            lines.append(f"  Details: {self.details}")
        return "\n".join(lines)


# ============================================================================
# CONCURRENCE
# ============================================================================

def concurrence_pure(
    state: Union[Ket, np.ndarray],
    dims: Optional[List[int]] = None
) -> float:
    """
    Compute concurrence for a pure state
    
    For a pure state |ψ⟩, concurrence = 2 * sqrt(det(ρ_A))
    where ρ_A is the reduced density matrix.
    
    Args:
        state: Pure state (Ket or array)
        dims: Subsystem dimensions [dim_A, dim_B] (default: [2, 2])
        
    Returns:
        float: Concurrence (0-1)
        
    Example:
        >>> from psiqit.quantum import bell_phi_plus
        >>> concurrence = concurrence_pure(bell_phi_plus())
        >>> print(concurrence)  # 1.0
    """
    if isinstance(state, Ket):
        state_data = state.data
    else:
        state_data = np.array(state, dtype=complex)
    
    # Default dimensions for bipartite systems
    if dims is None:
        dim = len(state_data)
        # Check if dimension is a power of 2
        n_qubits = int(np.log2(dim))
        if 2 ** n_qubits == dim:
            dim_a = 2 ** (n_qubits // 2)
            dim_b = dim // dim_a
            dims = [dim_a, dim_b]
        else:
            # Assume equal division
            dim_a = int(np.sqrt(dim))
            dim_b = dim // dim_a
            dims = [dim_a, dim_b]
    
    if len(dims) != 2:
        raise ValueError("Concurrence currently only supports bipartite systems")
    
    dim_a, dim_b = dims
    
    # Compute reduced density matrix
    rho = np.outer(state_data, state_data.conj())
    rho_a = np.zeros((dim_a, dim_a), dtype=complex)
    
    # Partial trace over subsystem B
    for i in range(dim_a):
        for j in range(dim_a):
            for k in range(dim_b):
                idx_i = i * dim_b + k
                idx_j = j * dim_b + k
                rho_a[i, j] += rho[idx_i, idx_j]
    
    # Concurrence = sqrt(2 * (1 - Tr(ρ_A²)))
    purity_a = np.trace(rho_a @ rho_a).real
    concurrence = np.sqrt(max(0, 2 * (1 - purity_a)))
    
    return float(concurrence)


def concurrence_mixed(
    rho: Union[np.ndarray, Matrix],
    dims: Optional[List[int]] = None
) -> float:
    """
    Compute concurrence for a mixed state (Wootters formula)
    
    For two-qubit systems, concurrence = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    where λ_i are the square roots of eigenvalues of ρ(σ_y⊗σ_y)ρ*(σ_y⊗σ_y)
    
    Args:
        rho: Density matrix
        dims: Subsystem dimensions (default: [2, 2])
        
    Returns:
        float: Concurrence (0-1)
        
    Example:
        >>> rho = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
        >>> concurrence = concurrence_mixed(rho)
        >>> print(concurrence)  # 1.0
    """
    if isinstance(rho, Matrix):
        rho_data = rho.data
    else:
        rho_data = np.array(rho, dtype=complex)
    
    # Default to two-qubit system
    if dims is None:
        dims = [2, 2]
    
    if len(dims) != 2 or dims[0] != 2 or dims[1] != 2:
        raise ValueError("Mixed concurrence currently only supports 2-qubit systems")
    
    # Pauli-Y matrix
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_y_sigma_y = np.kron(sigma_y, sigma_y)
    
    # Compute R = ρ (σ_y⊗σ_y) ρ* (σ_y⊗σ_y)
    rho_conj = np.conj(rho_data)
    R = rho_data @ sigma_y_sigma_y @ rho_conj @ sigma_y_sigma_y
    
    # Eigenvalues of R
    eigvals = np.linalg.eigvals(R)
    eigvals = np.real(eigvals)
    eigvals = eigvals[eigvals > 1e-12]
    
    # Sort in descending order
    eigvals = np.sort(eigvals)[::-1]
    lambda_sqrt = np.sqrt(eigvals)
    
    # Concurrence = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    if len(lambda_sqrt) >= 4:
        concurrence = max(0, lambda_sqrt[0] - lambda_sqrt[1] - lambda_sqrt[2] - lambda_sqrt[3])
    else:
        # Not enough eigenvalues, use simplified formula
        concurrence = max(0, np.sum(lambda_sqrt) - 2 * np.max(lambda_sqrt))
    
    return float(min(concurrence, 1.0))


def concurrence(
    state_or_rho: Union[Ket, np.ndarray, Matrix],
    dims: Optional[List[int]] = None
) -> float:
    """
    Compute concurrence for a state (pure or mixed)
    
    Automatically detects if the input is pure or mixed.
    
    Args:
        state_or_rho: State vector or density matrix
        dims: Subsystem dimensions
        
    Returns:
        float: Concurrence (0-1)
        
    Example:
        >>> from psiqit.quantum import bell_phi_plus
        >>> concurrence(bell_phi_plus())  # 1.0
    """
    # Check if it's a density matrix (square matrix)
    if isinstance(state_or_rho, Matrix):
        data = state_or_rho.data
    else:
        data = np.array(state_or_rho, dtype=complex)
    
    if len(data.shape) == 2 and data.shape[0] == data.shape[1]:
        # It's a density matrix
        # Check if it's a pure state (rank 1)
        eigvals = np.linalg.eigvalsh(data)
        if np.sum(eigvals > 1e-10) <= 1:
            # Pure state, convert to ket
            # Get the dominant eigenvector
            eigvals, eigvecs = np.linalg.eigh(data)
            idx = np.argmax(eigvals)
            state = Ket(eigvecs[:, idx])
            return concurrence_pure(state, dims)
        else:
            return concurrence_mixed(data, dims)
    else:
        # It's a state vector
        return concurrence_pure(state_or_rho, dims)


# ============================================================================
# NEGATIVITY
# ============================================================================

def negativity(
    state_or_rho: Union[Ket, np.ndarray, Matrix],
    dims: Optional[List[int]] = None
) -> float:
    """
    Compute negativity of a quantum state
    
    N(ρ) = (||ρ^{T_A}||₁ - 1) / 2
    
    where ρ^{T_A} is the partial transpose with respect to subsystem A.
    
    Args:
        state_or_rho: State vector or density matrix
        dims: Subsystem dimensions
        
    Returns:
        float: Negativity (0-1)
        
    Example:
        >>> from psiqit.quantum import bell_phi_plus
        >>> neg = negativity(bell_phi_plus())
        >>> print(neg)  # 0.5
    """
    # Convert to density matrix
    if isinstance(state_or_rho, Ket) or (isinstance(state_or_rho, np.ndarray) and len(state_or_rho.shape) == 1):
        # Pure state
        if isinstance(state_or_rho, Ket):
            state_data = state_or_rho.data
        else:
            state_data = state_or_rho
        rho = np.outer(state_data, state_data.conj())
    else:
        # Density matrix
        if isinstance(state_or_rho, Matrix):
            rho = state_or_rho.data
        else:
            rho = np.array(state_or_rho, dtype=complex)
    
    # Default dimensions for bipartite systems
    if dims is None:
        dim = rho.shape[0]
        # Check if dimension is a power of 2
        n_qubits = int(np.log2(dim))
        if 2 ** n_qubits == dim:
            dim_a = 2 ** (n_qubits // 2)
            dim_b = dim // dim_a
            dims = [dim_a, dim_b]
        else:
            # Assume equal division
            dim_a = int(np.sqrt(dim))
            dim_b = dim // dim_a
            dims = [dim_a, dim_b]
    
    if len(dims) != 2:
        raise ValueError("Negativity currently only supports bipartite systems")
    
    dim_a, dim_b = dims
    
    # Partial transpose with respect to subsystem A
    rho_pt = np.zeros_like(rho)
    
    # Reshape to (dim_a, dim_b, dim_a, dim_b)
    rho_reshaped = rho.reshape(dim_a, dim_b, dim_a, dim_b)
    
    # Transpose subsystem A: (i, j, k, l) -> (k, j, i, l)
    rho_pt_reshaped = np.einsum('ijkl->kjil', rho_reshaped)
    
    # Reshape back
    rho_pt = rho_pt_reshaped.reshape(dim_a * dim_b, dim_a * dim_b)
    
    # Compute trace norm: ||ρ^{T_A}||₁ = sum of singular values
    singular_values = np.linalg.svd(rho_pt, compute_uv=False)
    trace_norm = np.sum(singular_values)
    
    # Negativity = (||ρ^{T_A}||₁ - 1) / 2
    neg = max(0, (trace_norm - 1) / 2)
    
    return float(neg)


def logarithmic_negativity(
    state_or_rho: Union[Ket, np.ndarray, Matrix],
    dims: Optional[List[int]] = None
) -> float:
    """
    Compute logarithmic negativity
    
    E_N(ρ) = log₂(||ρ^{T_A}||₁)
    
    Args:
        state_or_rho: State vector or density matrix
        dims: Subsystem dimensions
        
    Returns:
        float: Logarithmic negativity
        
    Example:
        >>> from psiqit.quantum import bell_phi_plus
        >>> log_neg = logarithmic_negativity(bell_phi_plus())
        >>> print(log_neg)  # 1.0
    """
    # Compute negativity
    neg = negativity(state_or_rho, dims)
    
    # Log negativity = log₂(2*neg + 1)
    log_neg = np.log2(2 * neg + 1)
    
    return float(log_neg)


# ============================================================================
# ENTANGLEMENT ENTROPY
# ============================================================================

def entanglement_entropy(
    state_or_rho: Union[Ket, np.ndarray, Matrix],
    dims: List[int],
    base: str = 'e'
) -> float:
    """
    Compute entanglement entropy (von Neumann entropy of reduced state)
    
    S_E(ρ) = S(ρ_A) = -Tr(ρ_A log ρ_A)
    
    Args:
        state_or_rho: State vector or density matrix
        dims: Subsystem dimensions [dim_A, dim_B]
        base: Logarithm base
        
    Returns:
        float: Entanglement entropy
        
    Example:
        >>> from psiqit.quantum import bell_phi_plus
        >>> S_E = entanglement_entropy(bell_phi_plus(), [2, 2])
        >>> print(S_E)  # 0.693147...
    """
    # Convert to density matrix
    if isinstance(state_or_rho, Ket) or (isinstance(state_or_rho, np.ndarray) and len(state_or_rho.shape) == 1):
        # Pure state
        if isinstance(state_or_rho, Ket):
            state_data = state_or_rho.data
        else:
            state_data = state_or_rho
        rho = np.outer(state_data, state_data.conj())
    else:
        # Density matrix
        if isinstance(state_or_rho, Matrix):
            rho = state_or_rho.data
        else:
            rho = np.array(state_or_rho, dtype=complex)
    
    if len(dims) != 2:
        raise ValueError("Entanglement entropy currently only supports bipartite systems")
    
    dim_a, dim_b = dims
    
    # Compute reduced density matrix
    rho_a = np.zeros((dim_a, dim_a), dtype=complex)
    
    for i in range(dim_a):
        for j in range(dim_a):
            for k in range(dim_b):
                idx_i = i * dim_b + k
                idx_j = j * dim_b + k
                rho_a[i, j] += rho[idx_i, idx_j]
    
    # Compute von Neumann entropy
    from .entropy import von_neumann_entropy
    return von_neumann_entropy(rho_a, base)


# ============================================================================
# SCHMIDT DECOMPOSITION
# ============================================================================

def schmidt_decomposition(
    state: Union[Ket, np.ndarray],
    dims: List[int]
) -> Dict[str, Any]:
    """
    Compute Schmidt decomposition of a bipartite pure state
    
    |ψ⟩ = Σ_i λ_i |i⟩_A ⊗ |i⟩_B
    
    Args:
        state: Pure state
        dims: Subsystem dimensions [dim_A, dim_B]
        
    Returns:
        Dict: Schmidt coefficients, Schmidt vectors, and rank
        
    Example:
        >>> from psiqit.quantum import bell_phi_plus
        >>> result = schmidt_decomposition(bell_phi_plus(), [2, 2])
        >>> print(result['coefficients'])  # [0.707, 0.707]
    """
    if isinstance(state, Ket):
        state_data = state.data
    else:
        state_data = np.array(state, dtype=complex)
    
    if len(dims) != 2:
        raise ValueError("Schmidt decomposition currently only supports bipartite systems")
    
    dim_a, dim_b = dims
    
    # Reshape state into a matrix
    # |ψ⟩ = Σ_{i,j} A_{i,j} |i⟩_A ⊗ |j⟩_B
    A = state_data.reshape(dim_a, dim_b)
    
    # Compute Schmidt decomposition: A = U Σ V†
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    
    # Schmidt coefficients (normalized)
    coeffs = S / np.sqrt(np.sum(S ** 2))
    
    # Schmidt vectors
    schmidt_a = [Ket(U[:, i]) for i in range(len(coeffs))]
    schmidt_b = [Ket(Vh[i, :].conj()) for i in range(len(coeffs))]
    
    logger.debug(f"Schmidt decomposition: rank={len(coeffs)}")
    
    return {
        'coefficients': [float(c) for c in coeffs],
        'vectors_a': schmidt_a,
        'vectors_b': schmidt_b,
        'rank': len(coeffs),
        'schmidt_rank': len(coeffs)
    }


def schmidt_rank(
    state: Union[Ket, np.ndarray],
    dims: List[int],
    tol: float = 1e-10
) -> int:
    """
    Compute Schmidt rank of a bipartite pure state
    
    Args:
        state: Pure state
        dims: Subsystem dimensions
        tol: Tolerance for zero coefficients
        
    Returns:
        int: Schmidt rank
        
    Example:
        >>> from psiqit.quantum import bell_phi_plus
        >>> rank = schmidt_rank(bell_phi_plus(), [2, 2])
        >>> print(rank)  # 2
    """
    result = schmidt_decomposition(state, dims)
    coeffs = result['coefficients']
    
    # Count coefficients above tolerance
    rank = sum(1 for c in coeffs if c > tol)
    
    return rank


# ============================================================================
# ENTANGLEMENT DETECTION
# ============================================================================

def is_entangled(
    state_or_rho: Union[Ket, np.ndarray, Matrix],
    dims: Optional[List[int]] = None,
    tol: float = 1e-10
) -> bool:
    """
    Detect if a quantum state is entangled
    
    Uses multiple criteria:
    1. For pure states: Schmidt rank > 1
    2. For mixed states: negativity > 0 (for bipartite)
    
    Args:
        state_or_rho: State vector or density matrix
        dims: Subsystem dimensions
        tol: Tolerance for numerical checks
        
    Returns:
        bool: True if entangled, False if separable
        
    Example:
        >>> from psiqit.quantum import bell_phi_plus, zero
        >>> is_entangled(bell_phi_plus())  # True
        >>> is_entangled(zero())  # False
    """
    # Check if it's a density matrix (square matrix)
    if isinstance(state_or_rho, Matrix):
        data = state_or_rho.data
    else:
        data = np.array(state_or_rho, dtype=complex)
    
    if len(data.shape) == 2 and data.shape[0] == data.shape[1]:
        # It's a density matrix
        # Use negativity criterion for bipartite systems
        
        # Default dimensions
        if dims is None:
            dim = data.shape[0]
            n_qubits = int(np.log2(dim))
            if 2 ** n_qubits == dim:
                dim_a = 2 ** (n_qubits // 2)
                dim_b = dim // dim_a
                dims = [dim_a, dim_b]
            else:
                # If not bipartite, try to determine
                # For simplicity, we'll assume it's a 2-qubit system if dim=4
                if dim == 4:
                    dims = [2, 2]
                else:
                    # Cannot determine, return False
                    logger.warning("Cannot determine entanglement for this density matrix")
                    return False
        
        # Check if it's a pure state (rank 1)
        eigvals = np.linalg.eigvalsh(data)
        if np.sum(eigvals > tol) <= 1:
            # Pure state: get the ket and check Schmidt rank
            eigvals, eigvecs = np.linalg.eigh(data)
            idx = np.argmax(eigvals)
            state = Ket(eigvecs[:, idx])
            return schmidt_rank(state, dims, tol) > 1
        else:
            # Mixed state: use negativity
            neg = negativity(data, dims)
            return neg > tol
    else:
        # It's a state vector
        # Default dimensions for bipartite systems
        if dims is None:
            dim = len(data)
            n_qubits = int(np.log2(dim))
            if 2 ** n_qubits == dim:
                dim_a = 2 ** (n_qubits // 2)
                dim_b = dim // dim_a
                dims = [dim_a, dim_b]
            else:
                # For arbitrary dimensions, try to split evenly
                import math
                dim_a = int(math.isqrt(dim))
                if dim_a * dim_a == dim:
                    dims = [dim_a, dim_a]
                else:
                    # Cannot determine
                    logger.warning("Cannot determine subsystem dimensions for this state")
                    return False
        
        # Check Schmidt rank
        return schmidt_rank(data, dims, tol) > 1


# ============================================================================
# ADDITIONAL ENTANGLEMENT MEASURES
# ============================================================================

def entanglement_of_formation(
    state_or_rho: Union[Ket, np.ndarray, Matrix],
    dims: Optional[List[int]] = None
) -> float:
    """
    Compute entanglement of formation for two-qubit states
    
    E_F(ρ) = h((1 + √(1 - C²))/2)
    where h(x) = -x log₂(x) - (1-x) log₂(1-x)
    
    Args:
        state_or_rho: State vector or density matrix
        dims: Subsystem dimensions
        
    Returns:
        float: Entanglement of formation (0-1)
        
    Example:
        >>> from psiqit.quantum import bell_phi_plus
        >>> EOF = entanglement_of_formation(bell_phi_plus())
        >>> print(EOF)  # 1.0
    """
    # Compute concurrence
    C = concurrence(state_or_rho, dims)
    
    # Compute binary entropy
    def h(x):
        if x <= 0 or x >= 1:
            return 0.0
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)
    
    # Entanglement of formation
    x = (1 + np.sqrt(1 - C**2)) / 2
    EOF = h(x)
    
    return float(EOF)


def distillable_entanglement(
    state_or_rho: Union[Ket, np.ndarray, Matrix],
    dims: Optional[List[int]] = None
) -> float:
    """
    Compute distillable entanglement (upper bound using log negativity)
    
    E_D(ρ) ≤ log₂(||ρ^{T_A}||₁)
    
    Args:
        state_or_rho: State vector or density matrix
        dims: Subsystem dimensions
        
    Returns:
        float: Distillable entanglement (upper bound)
    """
    return logarithmic_negativity(state_or_rho, dims)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'EntanglementResult',
    'concurrence_pure',
    'concurrence_mixed',
    'concurrence',
    'negativity',
    'logarithmic_negativity',
    'entanglement_entropy',
    'schmidt_decomposition',
    'schmidt_rank',
    'is_entangled',
    'entanglement_of_formation',
    'distillable_entanglement',
]