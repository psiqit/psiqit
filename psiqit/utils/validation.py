# psiqit/utils/validation.py

"""
Validation Utilities Module
Validate quantum states, matrices, and circuits
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from ..quantum.state import Ket
from ..quantum.operator import Operator
from ..math.qalgebra import Matrix, Vector
from ..utils.logger import logger

# ============================================================================
# VALIDATION RESULT CLASS
# ============================================================================

@dataclass
class ValidationResult:
    """
    Result container for validation operations
    
    Attributes:
        is_valid: Whether validation passed
        message: Additional message
        details: Additional details
    """
    is_valid: bool = True
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "Valid" if self.is_valid else "Invalid"
        return f"ValidationResult(status={status}, message={self.message})"
    
    def __str__(self) -> str:
        lines = [
            "Validation Result:",
            f"  Valid: {self.is_valid}",
            f"  Message: {self.message}",
        ]
        if self.details:
            lines.append(f"  Details: {self.details}")
        return "\n".join(lines)
    
    def __bool__(self) -> bool:
        return self.is_valid


# ============================================================================
# MATRIX VALIDATION
# ============================================================================

def is_unitary(
    matrix: Union[np.ndarray, List[List], Operator, Matrix],
    tol: float = 1e-10
) -> ValidationResult:
    """
    Check if a matrix is unitary: U†U = I
    
    Args:
        matrix: Matrix to validate
        tol: Tolerance for numerical checks
        
    Returns:
        ValidationResult: Validation result
        
    Example:
        >>> from psiqit.quantum import pauli_x
        >>> result = is_unitary(pauli_x())
        >>> print(result.is_valid)  # True
    """
    if isinstance(matrix, Operator):
        data = matrix.data
    elif isinstance(matrix, Matrix):
        data = matrix.data
    else:
        data = np.array(matrix, dtype=complex)
    
    # Check if square
    if data.shape[0] != data.shape[1]:
        return ValidationResult(
            is_valid=False,
            message="Matrix is not square",
            details={'shape': data.shape}
        )
    
    dim = data.shape[0]
    identity = np.eye(dim, dtype=complex)
    
    # Check U†U = I
    product = data.conj().T @ data
    is_valid = np.allclose(product, identity, atol=tol)
    
    if not is_valid:
        diff = np.max(np.abs(product - identity))
        return ValidationResult(
            is_valid=False,
            message=f"U†U ≠ I (max difference: {diff:.2e})",
            details={'max_difference': float(diff)}
        )
    
    return ValidationResult(
        is_valid=True,
        message="Matrix is unitary",
        details={'dimension': dim}
    )


def is_hermitian(
    matrix: Union[np.ndarray, List[List], Operator, Matrix],
    tol: float = 1e-10
) -> ValidationResult:
    """
    Check if a matrix is Hermitian: H† = H
    
    Args:
        matrix: Matrix to validate
        tol: Tolerance for numerical checks
        
    Returns:
        ValidationResult: Validation result
        
    Example:
        >>> from psiqit.quantum import pauli_z
        >>> result = is_hermitian(pauli_z())
        >>> print(result.is_valid)  # True
    """
    if isinstance(matrix, Operator):
        data = matrix.data
    elif isinstance(matrix, Matrix):
        data = matrix.data
    else:
        data = np.array(matrix, dtype=complex)
    
    # Check if square
    if data.shape[0] != data.shape[1]:
        return ValidationResult(
            is_valid=False,
            message="Matrix is not square",
            details={'shape': data.shape}
        )
    
    # Check H† = H
    diff = data.conj().T - data
    max_diff = np.max(np.abs(diff))
    is_valid = max_diff < tol
    
    if not is_valid:
        return ValidationResult(
            is_valid=False,
            message=f"H† ≠ H (max difference: {max_diff:.2e})",
            details={'max_difference': float(max_diff)}
        )
    
    return ValidationResult(
        is_valid=True,
        message="Matrix is Hermitian",
        details={'dimension': data.shape[0]}
    )


def is_positive_semidefinite(
    matrix: Union[np.ndarray, List[List], Operator, Matrix],
    tol: float = 1e-10
) -> ValidationResult:
    """
    Check if a matrix is positive semidefinite: all eigenvalues >= 0
    
    Args:
        matrix: Matrix to validate
        tol: Tolerance for numerical checks
        
    Returns:
        ValidationResult: Validation result
        
    Example:
        >>> rho = np.array([[0.5, 0], [0, 0.5]])
        >>> result = is_positive_semidefinite(rho)
        >>> print(result.is_valid)  # True
    """
    if isinstance(matrix, Operator):
        data = matrix.data
    elif isinstance(matrix, Matrix):
        data = matrix.data
    else:
        data = np.array(matrix, dtype=complex)
    
    # Check if square
    if data.shape[0] != data.shape[1]:
        return ValidationResult(
            is_valid=False,
            message="Matrix is not square",
            details={'shape': data.shape}
        )
    
    # Check if Hermitian
    hermitian_check = is_hermitian(data, tol)
    if not hermitian_check.is_valid:
        return ValidationResult(
            is_valid=False,
            message="Matrix is not Hermitian",
            details={'hermitian_message': hermitian_check.message}
        )
    
    # Check eigenvalues
    eigvals = np.linalg.eigvalsh(data)
    min_eig = np.min(eigvals)
    is_valid = min_eig >= -tol
    
    if not is_valid:
        return ValidationResult(
            is_valid=False,
            message=f"Negative eigenvalue: {min_eig:.2e}",
            details={'min_eigenvalue': float(min_eig)}
        )
    
    return ValidationResult(
        is_valid=True,
        message="Matrix is positive semidefinite",
        details={
            'dimension': data.shape[0],
            'min_eigenvalue': float(min_eig)
        }
    )


def is_density_matrix(
    matrix: Union[np.ndarray, List[List], Operator, Matrix],
    tol: float = 1e-10
) -> ValidationResult:
    """
    Check if a matrix is a valid density matrix:
    - Hermitian
    - Positive semidefinite
    - Trace = 1
    
    Args:
        matrix: Matrix to validate
        tol: Tolerance for numerical checks
        
    Returns:
        ValidationResult: Validation result
        
    Example:
        >>> rho = np.array([[0.5, 0], [0, 0.5]])
        >>> result = is_density_matrix(rho)
        >>> print(result.is_valid)  # True
    """
    if isinstance(matrix, Operator):
        data = matrix.data
    elif isinstance(matrix, Matrix):
        data = matrix.data
    else:
        data = np.array(matrix, dtype=complex)
    
    # Check if square
    if data.shape[0] != data.shape[1]:
        return ValidationResult(
            is_valid=False,
            message="Matrix is not square",
            details={'shape': data.shape}
        )
    
    # Check Hermitian
    hermitian_check = is_hermitian(data, tol)
    if not hermitian_check.is_valid:
        return ValidationResult(
            is_valid=False,
            message="Matrix is not Hermitian",
            details={'hermitian_message': hermitian_check.message}
        )
    
    # Check positive semidefinite
    psd_check = is_positive_semidefinite(data, tol)
    if not psd_check.is_valid:
        return ValidationResult(
            is_valid=False,
            message="Matrix is not positive semidefinite",
            details={'psd_message': psd_check.message}
        )
    
    # Check trace = 1
    trace_val = np.trace(data).real
    is_valid = abs(trace_val - 1.0) < tol
    
    if not is_valid:
        return ValidationResult(
            is_valid=False,
            message=f"Trace = {trace_val:.6f} ≠ 1",
            details={'trace': float(trace_val)}
        )
    
    # Compute purity
    purity = np.trace(data @ data).real
    
    return ValidationResult(
        is_valid=True,
        message="Valid density matrix",
        details={
            'dimension': data.shape[0],
            'trace': float(trace_val),
            'purity': float(purity)
        }
    )


# ============================================================================
# STATE VALIDATION
# ============================================================================

def is_normalized(
    state: Union[Ket, np.ndarray, List],
    tol: float = 1e-10
) -> ValidationResult:
    """
    Check if a state is normalized: ⟨ψ|ψ⟩ = 1
    
    Args:
        state: Quantum state
        tol: Tolerance for numerical checks
        
    Returns:
        ValidationResult: Validation result
        
    Example:
        >>> from psiqit.quantum import zero
        >>> result = is_normalized(zero())
        >>> print(result.is_valid)  # True
    """
    if isinstance(state, Ket):
        data = state.data
    else:
        data = np.array(state, dtype=complex)
    
    norm = np.linalg.norm(data)
    is_valid = abs(norm - 1.0) < tol
    
    if not is_valid:
        return ValidationResult(
            is_valid=False,
            message=f"Norm = {norm:.6f} ≠ 1",
            details={'norm': float(norm)}
        )
    
    return ValidationResult(
        is_valid=True,
        message="State is normalized",
        details={'norm': float(norm), 'dimension': len(data)}
    )


def are_orthogonal(
    state1: Union[Ket, np.ndarray, List],
    state2: Union[Ket, np.ndarray, List],
    tol: float = 1e-10
) -> ValidationResult:
    """
    Check if two states are orthogonal: ⟨ψ|φ⟩ = 0
    
    Args:
        state1: First state
        state2: Second state
        tol: Tolerance for numerical checks
        
    Returns:
        ValidationResult: Validation result
        
    Example:
        >>> from psiqit.quantum import zero, one
        >>> result = are_orthogonal(zero(), one())
        >>> print(result.is_valid)  # True
    """
    if isinstance(state1, Ket):
        data1 = state1.data
    else:
        data1 = np.array(state1, dtype=complex)
    
    if isinstance(state2, Ket):
        data2 = state2.data
    else:
        data2 = np.array(state2, dtype=complex)
    
    if len(data1) != len(data2):
        return ValidationResult(
            is_valid=False,
            message=f"Dimension mismatch: {len(data1)} vs {len(data2)}",
            details={'dim1': len(data1), 'dim2': len(data2)}
        )
    
    overlap = np.vdot(data1, data2)
    is_valid = abs(overlap) < tol
    
    if not is_valid:
        return ValidationResult(
            is_valid=False,
            message=f"Overlap = {abs(overlap):.6f} ≠ 0",
            details={'overlap': float(abs(overlap))}
        )
    
    return ValidationResult(
        is_valid=True,
        message="States are orthogonal",
        details={'dimension': len(data1)}
    )


def are_identical(
    state1: Union[Ket, np.ndarray, List],
    state2: Union[Ket, np.ndarray, List],
    ignore_phase: bool = True,
    tol: float = 1e-10
) -> ValidationResult:
    """
    Check if two states are identical (up to global phase)
    
    Args:
        state1: First state
        state2: Second state
        ignore_phase: Ignore global phase difference
        tol: Tolerance for numerical checks
        
    Returns:
        ValidationResult: Validation result
        
    Example:
        >>> from psiqit.quantum import zero, one
        >>> result = are_identical(zero(), one())
        >>> print(result.is_valid)  # False
    """
    if isinstance(state1, Ket):
        data1 = state1.data
    else:
        data1 = np.array(state1, dtype=complex)
    
    if isinstance(state2, Ket):
        data2 = state2.data
    else:
        data2 = np.array(state2, dtype=complex)
    
    if len(data1) != len(data2):
        return ValidationResult(
            is_valid=False,
            message=f"Dimension mismatch: {len(data1)} vs {len(data2)}",
            details={'dim1': len(data1), 'dim2': len(data2)}
        )
    
    # Normalize if needed
    norm1 = np.linalg.norm(data1)
    norm2 = np.linalg.norm(data2)
    if norm1 > 0:
        data1 = data1 / norm1
    if norm2 > 0:
        data2 = data2 / norm2
    
    if ignore_phase:
        # Check up to global phase
        # Find phase that aligns state1 to state2
        overlap = np.vdot(data1, data2)
        phase = np.angle(overlap)
        data1_aligned = data1 * np.exp(1j * phase)
        is_valid = np.allclose(data1_aligned, data2, atol=tol)
    else:
        is_valid = np.allclose(data1, data2, atol=tol)
    
    if not is_valid:
        return ValidationResult(
            is_valid=False,
            message="States are not identical",
            details={'max_difference': float(np.max(np.abs(data1 - data2)))}
        )
    
    return ValidationResult(
        is_valid=True,
        message="States are identical",
        details={'dimension': len(data1), 'ignore_phase': ignore_phase}
    )


def fidelity(
    state1: Union[Ket, np.ndarray, List],
    state2: Union[Ket, np.ndarray, List]
) -> float:
    """
    Compute the fidelity between two states: F = |⟨ψ|φ⟩|²
    
    Args:
        state1: First state
        state2: Second state
        
    Returns:
        float: Fidelity (0-1)
        
    Example:
        >>> from psiqit.quantum import zero, plus
        >>> fid = fidelity(zero(), plus())
        >>> print(fid)  # 0.5
    """
    if isinstance(state1, Ket):
        data1 = state1.data
    else:
        data1 = np.array(state1, dtype=complex)
    
    if isinstance(state2, Ket):
        data2 = state2.data
    else:
        data2 = np.array(state2, dtype=complex)
    
    # Normalize
    norm1 = np.linalg.norm(data1)
    norm2 = np.linalg.norm(data2)
    if norm1 > 0:
        data1 = data1 / norm1
    if norm2 > 0:
        data2 = data2 / norm2
    
    overlap = np.vdot(data1, data2)
    return float(abs(overlap) ** 2)


# ============================================================================
# CIRCUIT VALIDATION
# ============================================================================

def validate_circuit(
    circuit: 'QuantumCircuit'
) -> ValidationResult:
    """
    Validate a quantum circuit
    
    Args:
        circuit: QuantumCircuit object
        
    Returns:
        ValidationResult: Validation result
        
    Example:
        >>> from psiqit.circuits import QuantumCircuit
        >>> circ = QuantumCircuit(2)
        >>> circ.h(0).cx(0, 1)
        >>> result = validate_circuit(circ)
        >>> print(result.is_valid)  # True
    """
    try:
        # Check n_qubits
        if not hasattr(circuit, 'n_qubits'):
            return ValidationResult(
                is_valid=False,
                message="Circuit missing n_qubits attribute"
            )
        
        n_qubits = circuit.n_qubits
        if n_qubits < 1:
            return ValidationResult(
                is_valid=False,
                message=f"Invalid number of qubits: {n_qubits}"
            )
        
        # Check gates
        gates = circuit.get_gates() if hasattr(circuit, 'get_gates') else []
        
        # Check if all gate qubits are valid
        for gate in gates:
            qubits = gate.get('qubits', [])
            for q in qubits:
                if q < 0 or q >= n_qubits:
                    return ValidationResult(
                        is_valid=False,
                        message=f"Invalid qubit index {q} in gate {gate.get('name', 'unknown')}"
                    )
        
        # Check depth
        depth = circuit.depth if hasattr(circuit, 'depth') else 0
        
        return ValidationResult(
            is_valid=True,
            message="Circuit is valid",
            details={
                'n_qubits': n_qubits,
                'gates': len(gates),
                'depth': depth
            }
        )
        
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            message=f"Validation error: {str(e)}"
        )


# ============================================================================
# ADDITIONAL VALIDATION UTILITIES
# ============================================================================

def is_valid_qubit_index(index: int, n_qubits: int) -> bool:
    """
    Check if a qubit index is valid
    
    Args:
        index: Qubit index
        n_qubits: Number of qubits
        
    Returns:
        bool: True if valid, False otherwise
    """
    return 0 <= index < n_qubits


def is_power_of_two(n: int) -> bool:
    """
    Check if a number is a power of two
    
    Args:
        n: Number to check
        
    Returns:
        bool: True if power of two, False otherwise
        
    Example:
        >>> is_power_of_two(8)  # True
        >>> is_power_of_two(10)  # False
    """
    return n > 0 and (n & (n - 1)) == 0


def validate_qubits(n_qubits: int) -> None:
    """
    Validate number of qubits (raises ValueError if invalid)
    
    Args:
        n_qubits: Number of qubits
        
    Raises:
        ValueError: If n_qubits is invalid
    """
    if not isinstance(n_qubits, int):
        raise TypeError(f"n_qubits must be int, got {type(n_qubits)}")
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if n_qubits > 30:
        raise ValueError(f"n_qubits must be <= 30 (simulation limit), got {n_qubits}")


def validate_dimension(dim: int) -> None:
    """
    Validate Hilbert space dimension (raises ValueError if invalid)
    
    Args:
        dim: Hilbert space dimension
        
    Raises:
        ValueError: If dim is invalid
    """
    if not isinstance(dim, int):
        raise TypeError(f"dim must be int, got {type(dim)}")
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Result class
    'ValidationResult',
    
    # Matrix validation
    'is_unitary',
    'is_hermitian',
    'is_positive_semidefinite',
    'is_density_matrix',
    
    # State validation
    'is_normalized',
    'are_orthogonal',
    'are_identical',
    'fidelity',
    
    # Circuit validation
    'validate_circuit',
    
    # Additional utilities
    'is_valid_qubit_index',
    'is_power_of_two',
    'validate_qubits',
    'validate_dimension',
] 