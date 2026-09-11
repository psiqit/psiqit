#psiqit/circuits/qubit.py

import numpy as np
from typing import Optional, Union, List
from ..quantum.state import Ket, zero, one, plus, minus, random_state
from ..utils.logger import logger
from ..utils.validation import validate_qubits


class Qubit:
    """
    Single qubit representation with state management
    
    A qubit is the fundamental unit of quantum information.
    It can be in state |0⟩, |1⟩, or any superposition of these.
    
    Example:
        >>> from psiqit.circuits import Qubit
        >>> q = Qubit(0)
        >>> print(q.state)  # |0⟩
        >>> q.state = [1/np.sqrt(2), 1/np.sqrt(2)]  # |+⟩
        >>> result = q.measure()
        >>> print(result)  # 0 or 1
    """
    
    def __init__(self, index: int, initial_state: Optional[Union[List, np.ndarray, Ket]] = None):
        """
        Initialize a qubit
        
        Args:
            index: Qubit index (0-based)
            initial_state: Initial state (default: |0⟩)
                          Can be list, numpy array, or Ket object
        
        Example:
            >>> q = Qubit(0)  # |0⟩
            >>> q = Qubit(1, [1, 0])  # |0⟩
            >>> q = Qubit(2, plus())  # |+⟩
        """
        if index < 0:
            raise ValueError(f"Qubit index must be >= 0, got {index}")
        
        self._index = index
        self._measured = False
        self._value = 0
        
        # Set initial state
        if initial_state is None:
            self._state = zero()
        elif isinstance(initial_state, Ket):
            self._state = initial_state.copy()
        else:
            # Convert list/numpy to Ket
            self._state = Ket(np.array(initial_state, dtype=complex))
        
        # Ensure correct dimension (2 for qubit)
        if self._state.dim != 2:
            raise ValueError(f"Qubit state must be 2-dimensional, got dim={self._state.dim}")
        
        logger.debug(f"Qubit {index} initialized with state: {self._state}")
    
    @property
    def index(self) -> int:
        """Get the qubit index"""
        return self._index
    
    @property
    def state(self) -> Ket:
        """
        Get the current quantum state of the qubit
        
        Returns:
            Ket: Current state vector
        
        Example:
            >>> q = Qubit(0)
            >>> print(q.state)  # |0⟩
        """
        return self._state
    
    @state.setter
    def state(self, new_state: Union[List, np.ndarray, Ket]):
        """
        Set the quantum state of the qubit
        
        Args:
            new_state: New state (list, numpy array, or Ket)
        
        Example:
            >>> q = Qubit(0)
            >>> q.state = [1/np.sqrt(2), 1/np.sqrt(2)]  # Set to |+⟩
        """
        if isinstance(new_state, Ket):
            if new_state.dim != 2:
                raise ValueError(f"Qubit state must be 2-dimensional, got dim={new_state.dim}")
            self._state = new_state.copy()
        else:
            state = Ket(np.array(new_state, dtype=complex))
            if state.dim != 2:
                raise ValueError(f"Qubit state must be 2-dimensional, got dim={state.dim}")
            self._state = state
        
        # Reset measurement status
        self._measured = False
        self._value = 0
        
        logger.debug(f"Qubit {self._index} state set to: {self._state}")
    
    @property
    def is_measured(self) -> bool:
        """
        Check if the qubit has been measured
        
        Returns:
            bool: True if measured, False otherwise
        
        Example:
            >>> q = Qubit(0)
            >>> print(q.is_measured)  # False
            >>> q.measure()
            >>> print(q.is_measured)  # True
        """
        return self._measured
    
    @property
    def value(self) -> Optional[int]:
        """
        Get the measurement value (if measured)
        
        Returns:
            Optional[int]: 0 or 1 if measured, None otherwise
        
        Example:
            >>> q = Qubit(0)
            >>> print(q.value)  # None
            >>> q.measure()
            >>> print(q.value)  # 0 or 1
        """
        return self._value if self._measured else None
    
    def measure(self) -> int:
        """
        Measure the qubit in the computational basis (Z-basis)
        
        The qubit collapses to |0⟩ or |1⟩ with probabilities
        determined by the Born rule.
        
        Returns:
            int: Measurement outcome (0 or 1)
        
        Example:
            >>> q = Qubit(0)
            >>> q.state = [1/np.sqrt(2), 1/np.sqrt(2)]  # |+⟩
            >>> result = q.measure()  # 50% chance of 0, 50% chance of 1
            >>> print(result)
        """
        # Calculate probabilities
        probs = np.abs(self._state.data) ** 2
        probs = probs / np.sum(probs)  # Normalize
        
        # Sample
        outcome = int(np.random.choice([0, 1], p=probs))
        
        # Collapse state to the measured outcome
        if outcome == 0:
            self._state = zero()
        else:
            self._state = one()
        
        self._measured = True
        self._value = outcome
        
        logger.debug(f"Qubit {self._index} measured: {outcome}")
        
        return outcome
    
    def reset(self) -> 'Qubit':
        """
        Reset the qubit to |0⟩ state
        
        Returns:
            Qubit: Self for chaining
        
        Example:
            >>> q = Qubit(0)
            >>> q.state = [0, 1]  # |1⟩
            >>> q.reset()
            >>> print(q.state)  # |0⟩
        """
        self._state = zero()
        self._measured = False
        self._value = 0
        
        logger.debug(f"Qubit {self._index} reset to |0⟩")
        
        return self
    
    def apply_gate(self, gate_matrix: np.ndarray) -> 'Qubit':
        """
        Apply a gate to the qubit
        
        Args:
            gate_matrix: 2x2 unitary matrix
        
        Returns:
            Qubit: Self for chaining
        
        Example:
            >>> from psiqit.quantum import pauli_x
            >>> q = Qubit(0)
            >>> q.apply_gate(pauli_x().data)  # Apply X gate (flip)
        """
        if gate_matrix.shape != (2, 2):
            raise ValueError(f"Gate matrix must be 2x2, got shape {gate_matrix.shape}")
        
        # Apply gate
        new_state = gate_matrix @ self._state.data
        self._state = Ket(new_state)
        
        # Reset measurement status
        self._measured = False
        self._value = 0
        
        logger.debug(f"Qubit {self._index} gate applied")
        
        return self
    
    def apply_x(self) -> 'Qubit':
        """Apply Pauli-X (NOT) gate"""
        from ..quantum.operator import pauli_x
        return self.apply_gate(pauli_x().data)
    
    def apply_y(self) -> 'Qubit':
        """Apply Pauli-Y gate"""
        from ..quantum.operator import pauli_y
        return self.apply_gate(pauli_y().data)
    
    def apply_z(self) -> 'Qubit':
        """Apply Pauli-Z gate"""
        from ..quantum.operator import pauli_z
        return self.apply_gate(pauli_z().data)
    
    def apply_h(self) -> 'Qubit':
        """Apply Hadamard gate"""
        from ..quantum.operator import hadamard
        return self.apply_gate(hadamard().data)
    
    def apply_rx(self, theta: float) -> 'Qubit':
        """Apply rotation around X-axis"""
        from ..quantum.operator import rx
        return self.apply_gate(rx(theta).data)
    
    def apply_ry(self, theta: float) -> 'Qubit':
        """Apply rotation around Y-axis"""
        from ..quantum.operator import ry
        return self.apply_gate(ry(theta).data)
    
    def apply_rz(self, theta: float) -> 'Qubit':
        """Apply rotation around Z-axis"""
        from ..quantum.operator import rz
        return self.apply_gate(rz(theta).data)
    
    def apply_phase(self, theta: float) -> 'Qubit':
        """Apply phase gate"""
        from ..quantum.operator import phase
        return self.apply_gate(phase(theta).data)
    
    def apply_s(self) -> 'Qubit':
        """Apply S gate (phase π/2)"""
        from ..quantum.operator import s_gate
        return self.apply_gate(s_gate().data)
    
    def apply_t(self) -> 'Qubit':
        """Apply T gate (phase π/4)"""
        from ..quantum.operator import t_gate
        return self.apply_gate(t_gate().data)
    
    def set_random(self, seed: Optional[int] = None) -> 'Qubit':
        """
        Set the qubit to a random state
        
        Args:
            seed: Random seed (optional)
        
        Returns:
            Qubit: Self for chaining
        
        Example:
            >>> q = Qubit(0)
            >>> q.set_random(seed=42)
            >>> print(q.state)  # Random state
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random complex amplitudes
        real = np.random.normal(0, 1, 2)
        imag = np.random.normal(0, 1, 2)
        state = (real + 1j * imag)
        
        # Normalize
        state = state / np.linalg.norm(state)
        
        self._state = Ket(state)
        self._measured = False
        self._value = 0
        
        logger.debug(f"Qubit {self._index} set to random state")
        
        return self
    
    def set_zero(self) -> 'Qubit':
        """Set the qubit to |0⟩ state"""
        self._state = zero()
        self._measured = False
        self._value = 0
        return self
    
    def set_one(self) -> 'Qubit':
        """Set the qubit to |1⟩ state"""
        self._state = one()
        self._measured = False
        self._value = 0
        return self
    
    def set_plus(self) -> 'Qubit':
        """Set the qubit to |+⟩ state"""
        self._state = plus()
        self._measured = False
        self._value = 0
        return self
    
    def set_minus(self) -> 'Qubit':
        """Set the qubit to |−⟩ state"""
        self._state = minus()
        self._measured = False
        self._value = 0
        return self
    
    def get_bloch_coordinates(self) -> tuple:
        """
        Get the Bloch sphere coordinates of the qubit state
        
        Returns:
            tuple: (x, y, z) coordinates on the Bloch sphere
        
        Example:
            >>> q = Qubit(0)
            >>> q.state = [1, 0]  # |0⟩
            >>> x, y, z = q.get_bloch_coordinates()
            >>> print(x, y, z)  # 0.0, 0.0, 1.0
        """
        # |ψ⟩ = a|0⟩ + b|1⟩
        a = self._state.data[0]
        b = self._state.data[1]
        
        # Bloch sphere coordinates:
        # x = 2 Re(b*a*)
        # y = 2 Im(b*a*)
        # z = |a|² - |b|²
        x = 2 * np.real(b * np.conj(a))
        y = 2 * np.imag(b * np.conj(a))
        z = np.abs(a)**2 - np.abs(b)**2
        
        return (float(x), float(y), float(z))
    
    def get_probability(self, state: int) -> float:
        """
        Get probability of measuring a specific state
        
        Args:
            state: 0 or 1
        
        Returns:
            float: Probability (0-1)
        
        Example:
            >>> q = Qubit(0)
            >>> q.state = [1/np.sqrt(2), 1/np.sqrt(2)]  # |+⟩
            >>> print(q.get_probability(0))  # 0.5
            >>> print(q.get_probability(1))  # 0.5
        """
        if state not in [0, 1]:
            raise ValueError(f"State must be 0 or 1, got {state}")
        
        return float(np.abs(self._state.data[state]) ** 2)
    
    def is_zero(self) -> bool:
        """Check if the qubit is in |0⟩ state"""
        return bool(np.allclose(self._state.data, zero().data))
    
    def is_one(self) -> bool:
        """Check if the qubit is in |1⟩ state"""
        return bool(np.allclose(self._state.data, one().data))
    
    def is_plus(self) -> bool:
        """Check if the qubit is in |+⟩ state"""
        return bool(np.allclose(self._state.data, plus().data))
    
    def is_minus(self) -> bool:
        """Check if the qubit is in |−⟩ state"""
        return bool(np.allclose(self._state.data, minus().data))
    
    def copy(self) -> 'Qubit':
        """
        Create a copy of the qubit
        
        Returns:
            Qubit: Copy of the qubit
        
        Example:
            >>> q1 = Qubit(0, [1, 0])
            >>> q2 = q1.copy()
            >>> print(q2.state)  # |0⟩
        """
        return Qubit(self._index, self._state.copy())
    
    # ============ Magic Methods ============
    
    def __repr__(self) -> str:
        """String representation"""
        status = "measured" if self._measured else "unmeasured"
        value_str = f"={self._value}" if self._measured else ""
        return f"Qubit(index={self._index}, state={self._state}, status={status}{value_str})"
    
    def __str__(self) -> str:
        """Pretty string representation"""
        status = "✓" if self._measured else "○"
        value_str = f" → {self._value}" if self._measured else ""
        return f"Q({self._index}){status}{value_str}: {self._state}"
    
    def __eq__(self, other: 'Qubit') -> bool:
        """Check equality of qubits"""
        if not isinstance(other, Qubit):
            return False
        return self._index == other._index and self._state == other._state
    
    def __ne__(self, other: 'Qubit') -> bool:
        """Check inequality of qubits"""
        return not self.__eq__(other)
    
    def __bool__(self) -> bool:
        """Boolean representation (True if measured and value=1)"""
        return self._measured and self._value == 1
    
    def __int__(self) -> int:
        """Integer representation (measurement value if measured)"""
        if not self._measured:
            raise ValueError("Cannot convert unmeasured qubit to int")
        return self._value
    
    def __float__(self) -> float:
        """Float representation (measurement value if measured)"""
        return float(int(self))
    
    def __format__(self, format_spec: str) -> str:
        """Format string representation"""
        if format_spec == 's':
            return str(self)
        elif format_spec == 'r':
            return repr(self)
        else:
            return str(self)


# ============================================================================
# Helper functions
# ============================================================================

def create_qubits(n: int, initial_state: Optional[Union[List, np.ndarray]] = None) -> List[Qubit]:
    """
    Create a list of qubits
    
    Args:
        n: Number of qubits
        initial_state: Initial state for all qubits (default: |0⟩)
    
    Returns:
        List[Qubit]: List of qubits
    
    Example:
        >>> qubits = create_qubits(3)  # 3 qubits in |0⟩
        >>> qubits = create_qubits(2, [1, 0])  # 2 qubits in |0⟩
    """
    if n < 1:
        raise ValueError(f"Number of qubits must be >= 1, got {n}")
    
    return [Qubit(i, initial_state) for i in range(n)]


def qubit_from_bloch(x: float, y: float, z: float, index: int = 0) -> Qubit:
    """
    Create a qubit from Bloch sphere coordinates
    
    Args:
        x: x-coordinate
        y: y-coordinate
        z: z-coordinate
        index: Qubit index
    
    Returns:
        Qubit: Qubit with given Bloch coordinates
    
    Example:
        >>> q = qubit_from_bloch(0, 0, 1)  # |0⟩
        >>> q = qubit_from_bloch(1, 0, 0)  # |+⟩
    """
    # Normalize if needed
    r = np.sqrt(x**2 + y**2 + z**2)
    if r > 1e-10:
        x, y, z = x/r, y/r, z/r
    
    # Convert Bloch to state vector
    # |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
    theta = np.arccos(z)
    phi = np.arctan2(y, x) if x**2 + y**2 > 1e-10 else 0
    
    a = np.cos(theta/2)
    b = np.exp(1j * phi) * np.sin(theta/2)
    
    return Qubit(index, [a, b])


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'Qubit',
    'create_qubits',
    'qubit_from_bloch',
]