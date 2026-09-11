#psiqit/quantum/state.py

"""
Quantum States Module
Ket, Bra, and various quantum state representations
"""

import numpy as np
from typing import List, Union, Optional, Tuple, Dict, Any
# فقط PI, SQRT2, SQRT3 را از math.qalgebra import می‌کنیم (بدون وابستگی دایره‌ای)
from ..math.qalgebra import PI, SQRT2, SQRT3


# ============================================================================
# KET CLASS
# ============================================================================

class Ket:
    """
    Ket vector representation |ψ⟩
    
    Example:
        >>> from psiqit.quantum import Ket
        >>> ket = Ket([1, 0])  # |0⟩
        >>> ket = Ket([1/np.sqrt(2), 1/np.sqrt(2)])  # |+⟩
    """
    
    def __init__(self, data: Union[List[complex], np.ndarray], _normalized: bool = False):
        """
        Initialize a Ket state
        
        Args:
            data: Amplitude vector
            _normalized: If True, skip normalization (internal use)
        """
        self._data = np.array(data, dtype=complex)
        self._dim = len(self._data)
        
        if not _normalized:
            self.normalize()
        
        # Lazy import logger (برای جلوگیری از circular import)
        from ..utils.logger import logger
        logger.debug(f"Created Ket with dimension {self._dim}")
    
    @property
    def data(self) -> np.ndarray:
        """Return the state vector"""
        return self._data
    
    @property
    def dim(self) -> int:
        """Return the dimension of the Hilbert space"""
        return self._dim
    
    @property
    def is_normalized(self) -> bool:
        """Check if the state is normalized"""
        return np.isclose(np.linalg.norm(self._data), 1.0)
    
    def norm(self) -> float:
        """Return the norm of the state"""
        return float(np.linalg.norm(self._data))
    
    def normalize(self) -> 'Ket':
        """Normalize the state vector"""
        n = self.norm()
        if n > 0:
            self._data = self._data / n
        else:
            raise ValueError("Cannot normalize zero vector")
        return self
    
    def inner(self, other: 'Ket') -> complex:
        """
        Compute inner product ⟨ψ|φ⟩
        
        Example:
            >>> ket0 = Ket([1, 0])
            >>> ket1 = Ket([0, 1])
            >>> ket0.inner(ket1)  # 0j
        """
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return np.vdot(self._data, other.data)
    
    def outer(self, other: 'Ket') -> 'Operator':
        """
        Compute outer product |ψ⟩⟨φ|
        
        Example:
            >>> ket0 = Ket([1, 0])
            >>> ket1 = Ket([0, 1])
            >>> ket0.outer(ket1)  # [[0, 1], [0, 0]]
        """
        # Lazy import Operator برای جلوگیری از circular import
        from .operator import Operator
        return Operator(np.outer(self._data, np.conj(other.data)))
    
    def to_bra(self) -> 'Bra':
        """Convert to Bra vector ⟨ψ|"""
        return Bra(np.conj(self._data))
    
    def measure(self, shots: int = 1) -> Dict[str, Any]:
        """
        Measure the state in the computational basis
        
        Args:
            shots: Number of measurements
            
        Returns:
            Dictionary with counts and probabilities
        """
        probs = np.abs(self._data) ** 2
        probs = probs / np.sum(probs)
        
        # Sample
        outcomes = np.random.choice(range(self.dim), size=shots, p=probs)
        
        # Count outcomes
        bit_length = int(np.ceil(np.log2(self.dim)))
        counts = {}
        for out in outcomes:
            label = format(out, f'0{bit_length}b') if self.dim > 1 else str(out)
            counts[label] = counts.get(label, 0) + 1
        
        return {
            'counts': counts,
            'probabilities': probs.tolist(),
            'shots': shots,
            'dimension': self.dim
        }
    
    def prob(self, basis_state: int) -> float:
        """
        Get probability of a specific basis state
        
        Args:
            basis_state: Index of the basis state
            
        Returns:
            Probability (0-1)
        """
        if basis_state < 0 or basis_state >= self.dim:
            raise ValueError(f"Basis state {basis_state} out of range [0, {self.dim-1}]")
        return float(np.abs(self._data[basis_state]) ** 2)
    
    def sample(self, shots: int = 1) -> List[int]:
        """
        Sample measurement outcomes
        
        Args:
            shots: Number of samples
            
        Returns:
            List of measurement outcomes (integers)
        """
        probs = np.abs(self._data) ** 2
        probs = probs / np.sum(probs)
        return list(np.random.choice(range(self.dim), size=shots, p=probs))
    
    def copy(self) -> 'Ket':
        """Return a copy of the Ket state"""
        return Ket(self._data.copy(), _normalized=True)
    
    # ============ Magic Methods ============
    
    def __add__(self, other: 'Ket') -> 'Ket':
        """Add two Ket vectors"""
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return Ket(self._data + other.data)
    
    def __sub__(self, other: 'Ket') -> 'Ket':
        """Subtract two Ket vectors"""
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return Ket(self._data - other.data)
    
    def __mul__(self, scalar: complex) -> 'Ket':
        """Multiply by scalar"""
        return Ket(self._data * scalar)
    
    def __rmul__(self, scalar: complex) -> 'Ket':
        """Multiply by scalar (reverse)"""
        return self.__mul__(scalar)
    
    def __getitem__(self, index: int) -> complex:
        """Get amplitude at index"""
        return self._data[index]
    
    def __len__(self) -> int:
        """Return dimension"""
        return self.dim
    
    def __repr__(self) -> str:
        """String representation"""
        if self.dim <= 4:
            return f"Ket({self._data.tolist()})"
        return f"Ket(dim={self.dim})"
    
    def __str__(self) -> str:
        """Pretty string representation"""
        if self.dim == 2:
            return f"({self._data[0]:.3f})|0⟩ + ({self._data[1]:.3f})|1⟩"
        elif self.dim <= 4:
            terms = []
            for i, amp in enumerate(self._data):
                if abs(amp) > 1e-10:
                    terms.append(f"({amp:.3f})|{i}⟩")
            return " + ".join(terms) if terms else "0"
        else:
            bit_length = int(np.ceil(np.log2(self.dim)))
            terms = []
            for i, amp in enumerate(self._data[:10]):  # Show only first 10
                if abs(amp) > 1e-10:
                    binary = format(i, f'0{bit_length}b')
                    terms.append(f"({amp:.3f})|{binary}⟩")
            if len(self._data) > 10:
                terms.append("...")
            return " + ".join(terms) if terms else "0"
    
    def __eq__(self, other: 'Ket') -> bool:
        """Check equality (ignoring global phase)"""
        if not isinstance(other, Ket):
            return False
        if self.dim != other.dim:
            return False
        # Check if same up to global phase
        inner_product = self.inner(other)
        return abs(abs(inner_product) - 1) < 1e-10


# ============================================================================
# BRA CLASS
# ============================================================================

class Bra:
    """
    Bra vector representation ⟨ψ|
    
    Example:
        >>> from psiqit.quantum import Ket, Bra
        >>> ket = Ket([1, 0])
        >>> bra = ket.to_bra()  # ⟨0|
    """
    
    def __init__(self, data: Union[List[complex], np.ndarray]):
        """Initialize a Bra vector"""
        self._data = np.array(data, dtype=complex)
        self._dim = len(self._data)
    
    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def __matmul__(self, ket: Ket) -> complex:
        """Inner product ⟨ψ|φ⟩"""
        if self.dim != ket.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {ket.dim}")
        return np.dot(self._data, ket.data)
    
    def to_ket(self) -> Ket:
        """Convert to Ket vector |ψ⟩"""
        return Ket(np.conj(self._data))
    
    def copy(self) -> 'Bra':
        """Return a copy of the Bra vector"""
        return Bra(self._data.copy())
    
    def __getitem__(self, index: int) -> complex:
        """Get amplitude at index"""
        return self._data[index]
    
    def __len__(self) -> int:
        """Return dimension"""
        return self.dim
    
    def __repr__(self) -> str:
        """String representation"""
        if self.dim <= 4:
            return f"Bra({self._data.tolist()})"
        return f"Bra(dim={self.dim})"
    
    def __str__(self) -> str:
        """Pretty string representation"""
        if self.dim == 2:
            return f"({np.conj(self._data[0]):.3f})⟨0| + ({np.conj(self._data[1]):.3f})⟨1|"
        else:
            terms = []
            for i, amp in enumerate(self._data):
                if abs(amp) > 1e-10:
                    terms.append(f"({np.conj(amp):.3f})⟨{i}|")
            return " + ".join(terms) if terms else "0"
    
    def __eq__(self, other: 'Bra') -> bool:
        """Check equality"""
        if not isinstance(other, Bra):
            return False
        return self.dim == other.dim and np.allclose(self._data, other._data)


# ============================================================================
# BASE FUNCTIONS
# ============================================================================

def ket(*amplitudes: complex) -> Ket:
    """
    Create a Ket state from amplitudes
    
    Example:
        >>> ket(1, 0)  # |0⟩
        >>> ket(1/np.sqrt(2), 1/np.sqrt(2))  # |+⟩
    """
    return Ket(list(amplitudes))


def basis(dim: int, index: int) -> Ket:
    """
    Create a basis state |index⟩ in dimension dim
    
    Args:
        dim: Hilbert space dimension
        index: Basis state index (0 to dim-1)
        
    Returns:
        Ket: Basis state
        
    Example:
        >>> basis(4, 2)  # |2⟩ in 4-dimensional space
    """
    if index < 0 or index >= dim:
        raise ValueError(f"Index {index} out of range [0, {dim-1}]")
    state = np.zeros(dim, dtype=complex)
    state[index] = 1.0
    return Ket(state, _normalized=True)


# ============================================================================
# SINGLE-QUBIT STATES
# ============================================================================

def zero() -> Ket:
    """|0⟩ state"""
    return Ket([1, 0], _normalized=True)


def one() -> Ket:
    """|1⟩ state"""
    return Ket([0, 1], _normalized=True)


def plus() -> Ket:
    """|+⟩ = (|0⟩ + |1⟩)/√2"""
    return Ket([1/np.sqrt(2), 1/np.sqrt(2)], _normalized=True)


def minus() -> Ket:
    """|−⟩ = (|0⟩ - |1⟩)/√2"""
    return Ket([1/np.sqrt(2), -1/np.sqrt(2)], _normalized=True)


def ip() -> Ket:
    """|i⟩ = (|0⟩ + i|1⟩)/√2"""
    return Ket([1/np.sqrt(2), 1j/np.sqrt(2)], _normalized=True)


def im() -> Ket:
    """|-i⟩ = (|0⟩ - i|1⟩)/√2"""
    return Ket([1/np.sqrt(2), -1j/np.sqrt(2)], _normalized=True)


# ============================================================================
# BELL STATES
# ============================================================================

def bell_phi_plus() -> Ket:
    """|Φ⁺⟩ = (|00⟩ + |11⟩)/√2"""
    return Ket([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], _normalized=True)


def bell_phi_minus() -> Ket:
    """|Φ⁻⟩ = (|00⟩ - |11⟩)/√2"""
    return Ket([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)], _normalized=True)


def bell_psi_plus() -> Ket:
    """|Ψ⁺⟩ = (|01⟩ + |10⟩)/√2"""
    return Ket([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], _normalized=True)


def bell_psi_minus() -> Ket:
    """|Ψ⁻⟩ = (|01⟩ - |10⟩)/√2"""
    return Ket([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], _normalized=True)


def bell_state(index: int) -> Ket:
    """
    Get Bell state by index (0-3)
    
    0: |Φ⁺⟩, 1: |Φ⁻⟩, 2: |Ψ⁺⟩, 3: |Ψ⁻⟩
    
    Example:
        >>> bell_state(0)  # |Φ⁺⟩
    """
    states = [bell_phi_plus, bell_phi_minus, bell_psi_plus, bell_psi_minus]
    if index < 0 or index >= 4:
        raise ValueError(f"Bell state index must be 0-3, got {index}")
    return states[index]()


# ============================================================================
# MULTI-QUBIT STATES
# ============================================================================

def ghz(n: int) -> Ket:
    """
    GHZ state: (|00...0⟩ + |11...1⟩)/√2
    
    Args:
        n: Number of qubits
        
    Returns:
        Ket: GHZ state
        
    Example:
        >>> ghz(3)  # (|000⟩ + |111⟩)/√2
    """
    # Lazy import validation برای جلوگیری از circular import
    from ..utils.validation import validate_qubits
    validate_qubits(n)
    dim = 2**n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1/np.sqrt(2)
    state[-1] = 1/np.sqrt(2)
    return Ket(state, _normalized=True)


def w_state(n: int) -> Ket:
    """
    W state: (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
    
    Args:
        n: Number of qubits
        
    Returns:
        Ket: W state
        
    Example:
        >>> w_state(3)  # (|100⟩ + |010⟩ + |001⟩)/√3
    """
    # Lazy import validation برای جلوگیری از circular import
    from ..utils.validation import validate_qubits
    validate_qubits(n)
    dim = 2**n
    state = np.zeros(dim, dtype=complex)
    for i in range(n):
        state[1 << i] = 1/np.sqrt(n)
    return Ket(state, _normalized=True)


# ============================================================================
# RANDOM STATES
# ============================================================================

def random_state(dim: int, seed: Optional[int] = None) -> Ket:
    """
    Generate a random quantum state
    
    Args:
        dim: Hilbert space dimension
        seed: Random seed (optional)
        
    Returns:
        Ket: Random normalized state
        
    Example:
        >>> random_state(4)  # Random 4-dimensional state
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random complex amplitudes
    real = np.random.normal(0, 1, dim)
    imag = np.random.normal(0, 1, dim)
    state = real + 1j * imag
    
    # Normalize
    state = state / np.linalg.norm(state)
    return Ket(state, _normalized=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_orthogonal(psi: Ket, phi: Ket, tol: float = 1e-10) -> bool:
    """Check if two states are orthogonal"""
    return abs(psi.inner(phi)) < tol


def is_same(psi: Ket, phi: Ket, tol: float = 1e-10) -> bool:
    """Check if two states are the same (up to global phase)"""
    inner = psi.inner(phi)
    return abs(abs(inner) - 1) < tol


def fidelity(psi: Ket, phi: Ket) -> float:
    """
    Calculate fidelity between two states: F = |⟨ψ|φ⟩|²
    
    Returns:
        float: Fidelity (0 to 1)
    """
    return float(abs(psi.inner(phi)) ** 2)


# ============================================================================
# CONTINUOUS VARIABLE STATES (Truncated)
# ============================================================================

def fock_state(n: int, n_levels: int) -> Ket:
    """
    Fock state |n⟩ in a truncated Hilbert space
    
    Args:
        n: Fock state number
        n_levels: Truncation level (Hilbert space dimension)
        
    Returns:
        Ket: Fock state
        
    Example:
        >>> fock_state(2, 10)  # |2⟩ in 10-dimensional space
    """
    if n >= n_levels:
        raise ValueError(f"Fock state {n} exceeds truncation {n_levels}")
    return basis(n_levels, n)


def coherent_state(alpha: complex, n_levels: int) -> Ket:
    """
    Coherent state |α⟩ in truncated Hilbert space
    
    |α⟩ = e^{-|α|²/2} Σ (α^n / √n!) |n⟩
    
    Args:
        alpha: Complex amplitude
        n_levels: Truncation level
        
    Returns:
        Ket: Coherent state
        
    Example:
        >>> coherent_state(0.5, 10)
    """
    from math import exp, sqrt, factorial
    
    state = np.zeros(n_levels, dtype=complex)
    norm = exp(-abs(alpha)**2 / 2)
    
    for n in range(n_levels):
        state[n] = norm * (alpha**n) / sqrt(factorial(n))
    
    return Ket(state, _normalized=True)


def squeezed_state(r: float, phi: float = 0.0, n_levels: int = 10) -> Ket:
    """
    Squeezed vacuum state in truncated Hilbert space
    
    |r,φ⟩ = sech(r) Σ √((2n)!)/(2^n n!) tanh(r)^n e^{inφ} |2n⟩
    
    Args:
        r: Squeezing parameter
        phi: Squeezing phase
        n_levels: Truncation level
        
    Returns:
        Ket: Squeezed state
        
    Example:
        >>> squeezed_state(0.5, 0, 10)
    """
    from math import sqrt, factorial, tanh, cosh
    
    state = np.zeros(n_levels, dtype=complex)
    
    # Only even Fock states are populated
    for n in range(n_levels // 2):
        idx = 2 * n
        if idx >= n_levels:
            break
        
        # Coefficient: √((2n)!)/(2^n n!) * tanh(r)^n * e^{inφ} / cosh(r)
        coef = sqrt(factorial(2*n)) / (2**n * factorial(n))
        coef *= tanh(r)**n * np.exp(1j * n * phi)
        coef /= np.sqrt(cosh(r))
        
        state[idx] = coef
    
    return Ket(state, _normalized=True)


def thermal_state(n_bar: float, n_levels: int) -> Ket:
    """
    Thermal state (not a pure state, but here we return the diagonal density
    matrix as a state vector of probabilities)
    
    Actually returns the pure state with thermal probabilities
    
    Args:
        n_bar: Average photon number
        n_levels: Truncation level
        
    Returns:
        Ket: State with thermal probabilities
    """
    state = np.zeros(n_levels, dtype=complex)
    
    for n in range(n_levels):
        # P(n) = n_bar^n / (1 + n_bar)^(n+1)
        prob = n_bar**n / (1 + n_bar)**(n+1)
        state[n] = np.sqrt(prob)
    
    return Ket(state, _normalized=True)


# ============================================================================
# MULTI-PARTY STATES
# ============================================================================

def alice_bell_state(index: int) -> Ket:
    """
    Bell state shared between Alice and Bob (2-qubit)
    
    Args:
        index: Bell state index (0-3)
        
    Returns:
        Ket: Bell state
    """
    return bell_state(index)


def alice_bob_ghz(n: int) -> Ket:
    """
    GHZ state shared between Alice and Bob
    
    Args:
        n: Number of qubits total
        
    Returns:
        Ket: GHZ state
    """
    return ghz(n)


def alice_bob_w(n: int) -> Ket:
    """
    W state shared between Alice and Bob
    
    Args:
        n: Number of qubits total
        
    Returns:
        Ket: W state
    """
    return w_state(n)


def ep_pair() -> Ket:
    """
    EPR pair (maximally entangled two-qubit state)
    Same as bell_phi_plus()
    
    Returns:
        Ket: EPR pair
    """
    return bell_phi_plus()


# ============================================================================
# ADDITIONAL STATES
# ============================================================================

def phase_state(theta: float, dim: int = 2) -> Ket:
    """
    Phase state: (|0⟩ + e^{iθ}|1⟩)/√2
    
    Args:
        theta: Phase angle
        dim: Dimension (2 for qubit)
        
    Returns:
        Ket: Phase state
        
    Example:
        >>> phase_state(np.pi/2)  # (|0⟩ + i|1⟩)/√2
    """
    if dim != 2:
        raise NotImplementedError("Phase state only implemented for dim=2")
    return Ket([1/np.sqrt(2), np.exp(1j * theta) / np.sqrt(2)], _normalized=True)


def dual_rail_qubit() -> Ket:
    """
    Dual-rail qubit representation
    
    This represents a single photon in two modes:
    |0⟩ = |1,0⟩, |1⟩ = |0,1⟩
    
    Returns:
        Ket: Dual-rail qubit state |1,0⟩
    """
    return Ket([1, 0], _normalized=True)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Classes
    'Ket',
    'Bra',
    
    # Base functions
    'ket',
    'basis',
    
    # Single-qubit states
    'zero',
    'one',
    'plus',
    'minus',
    'ip',
    'im',
    
    # Bell states
    'bell_phi_plus',
    'bell_phi_minus',
    'bell_psi_plus',
    'bell_psi_minus',
    'bell_state',
    
    # Multi-qubit states
    'ghz',
    'w_state',
    
    # Random states
    'random_state',
    
    # Utility functions
    'is_orthogonal',
    'is_same',
    'fidelity',
    
    # Continuous variable states
    'fock_state',
    'coherent_state',
    'squeezed_state',
    'thermal_state',
    
    # Multi-party states
    'alice_bell_state',
    'alice_bob_ghz',
    'alice_bob_w',
    'ep_pair',
    
    # Additional states
    'phase_state',
    'dual_rail_qubit',
]