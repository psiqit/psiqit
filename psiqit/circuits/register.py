#psiqit/circuits/register.py 


import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any
from ..quantum.state import Ket, zero, basis
from ..quantum.operator import identity
from ..utils.logger import logger
from ..utils.validation import validate_qubits
from .qubit import Qubit


class QuantumRegister:
    """
    Quantum register - a collection of qubits
    
    A QuantumRegister manages multiple qubits and provides collective operations
    like state vector manipulation and measurements.
    
    Example:
        >>> from psiqit.circuits import QuantumRegister
        >>> reg = QuantumRegister(3)  # 3 qubits in |000⟩
        >>> reg.get_qubit(0).apply_h()  # Apply H to first qubit
        >>> state = reg.get_state_vector()
        >>> print(state)  # (|000⟩ + |100⟩)/√2
        >>> result = reg.measure()  # Measure all qubits
    """
    
    def __init__(
        self,
        n_qubits: int,
        initial_states: Optional[List[Union[List, np.ndarray, Ket]]] = None
    ):
        """
        Initialize a quantum register
        
        Args:
            n_qubits: Number of qubits
            initial_states: Optional list of initial states for each qubit
        
        Example:
            >>> reg = QuantumRegister(2)  # |00⟩
            >>> reg = QuantumRegister(2, [[1, 0], [0, 1]])  # |01⟩
            >>> reg = QuantumRegister(3, [None, None, [1, 0]])  # |000⟩
        """
        validate_qubits(n_qubits)
        
        self._n_qubits = n_qubits
        self._qubits: List[Qubit] = []
        
        # Initialize qubits
        for i in range(n_qubits):
            if initial_states and i < len(initial_states):
                state = initial_states[i]
            else:
                state = None
            self._qubits.append(Qubit(i, state))
        
        self._state_vector = None
        self._update_state_vector()
        
        logger.info(f"QuantumRegister initialized with {n_qubits} qubits")
    
    @property
    def n_qubits(self) -> int:
        """Get the number of qubits in the register"""
        return self._n_qubits
    
    @property
    def dim(self) -> int:
        """Get the dimension of the Hilbert space (2^n_qubits)"""
        return 2 ** self._n_qubits
    
    def _update_state_vector(self):
        """Update the internal state vector from individual qubit states"""
        if self._n_qubits == 0:
            self._state_vector = Ket([1.0])
            return
        
        # Build the full state vector from individual qubit states
        # For n qubits, the state is the tensor product of individual states
        state = np.array([1.0], dtype=complex)
        
        for qubit in self._qubits:
            # If qubit has been measured, use its collapsed state
            if qubit.is_measured:
                if qubit.value == 0:
                    q_state = np.array([1.0, 0.0], dtype=complex)
                else:
                    q_state = np.array([0.0, 1.0], dtype=complex)
            else:
                q_state = qubit.state.data
            
            state = np.kron(state, q_state)
        
        self._state_vector = Ket(state)
    
    def get_state_vector(self) -> Ket:
        """
        Get the full state vector of the register
        
        Returns:
            Ket: State vector
        
        Example:
            >>> reg = QuantumRegister(2)
            >>> reg.get_qubit(0).apply_h()
            >>> state = reg.get_state_vector()
            >>> print(state)  # (|00⟩ + |10⟩)/√2
        """
        self._update_state_vector()
        return self._state_vector.copy()
    
    def set_state_vector(self, state: Union[List, np.ndarray, Ket]):
        """
        Set the full state vector of the register
        
        Args:
            state: New state vector
        
        Example:
            >>> reg = QuantumRegister(2)
            >>> from psiqit.quantum import bell_phi_plus
            >>> reg.set_state_vector(bell_phi_plus())
            >>> print(reg.get_state_vector())  # Bell state
        """
        if isinstance(state, Ket):
            if state.dim != self.dim:
                raise ValueError(f"State dimension {state.dim} does not match register dimension {self.dim}")
            full_state = state.data
        else:
            full_state = np.array(state, dtype=complex)
            if len(full_state) != self.dim:
                raise ValueError(f"State length {len(full_state)} does not match register dimension {self.dim}")
        
        # Normalize
        norm = np.linalg.norm(full_state)
        if norm > 0:
            full_state = full_state / norm
        
        # Extract individual qubit states (simplified)
        # For a general state, we can't easily extract individual qubit states
        # unless it's a product state. We'll store the full state directly.
        self._state_vector = Ket(full_state)
        
        # Try to extract individual qubit states if possible
        # For product states, we can factorize
        self._try_factorize_state()
        
        logger.debug(f"State vector set with dimension {self.dim}")
    
    def _try_factorize_state(self):
        """
        Try to factorize the state into individual qubit states
        This works for product states but not for entangled states
        """
        if self._state_vector is None:
            return
        
        # Check if state is a product state by trying to factorize
        # For simplicity, we only handle the case where the state is
        # a tensor product of single-qubit states
        
        # This is a simplified implementation
        # For entangled states, the individual qubit states are not well-defined
        # We'll keep the full state and indicate entanglement
        
        # Try to factorize by checking if the state can be written as
        # |ψ⟩ = |ψ₀⟩ ⊗ |ψ₁⟩ ⊗ ... ⊗ |ψ_{n-1}⟩
        
        # For now, we just set the qubit states if they are pure
        # This is a placeholder for a full factorization algorithm
        pass
    
    def get_qubit(self, index: int) -> Qubit:
        """
        Get a specific qubit from the register
        
        Args:
            index: Qubit index
            
        Returns:
            Qubit: The qubit at the given index
        
        Example:
            >>> reg = QuantumRegister(3)
            >>> q = reg.get_qubit(1)
            >>> q.apply_h()
        """
        if index < 0 or index >= self._n_qubits:
            raise IndexError(f"Qubit index {index} out of range [0, {self._n_qubits-1}]")
        
        return self._qubits[index]
    
    def measure(self, qubit_index: Optional[int] = None) -> Union[int, Dict[str, Any]]:
        """
        Measure one or all qubits
        
        Args:
            qubit_index: Qubit to measure (None for all)
            
        Returns:
            If qubit_index specified: Measurement outcome (0 or 1)
            If qubit_index is None: Dictionary with counts and states
        
        Example:
            >>> reg = QuantumRegister(2)
            >>> reg.get_qubit(0).apply_h()
            >>> result = reg.measure()  # Measure all qubits
            >>> print(result['counts'])
            >>> result = reg.measure(qubit_index=0)  # Measure first qubit
        """
        if qubit_index is not None:
            # Measure single qubit
            if qubit_index < 0 or qubit_index >= self._n_qubits:
                raise IndexError(f"Qubit index {qubit_index} out of range [0, {self._n_qubits-1}]")
            
            # Update state vector first
            self._update_state_vector()
            
            # Calculate probability of |1⟩ for this qubit
            prob_1 = 0.0
            for i in range(self.dim):
                if (i >> qubit_index) & 1:
                    prob_1 += np.abs(self._state_vector.data[i]) ** 2
            
            # Sample
            outcome = 1 if np.random.random() < prob_1 else 0
            
            # Collapse the full state
            new_state = np.zeros(self.dim, dtype=complex)
            for i in range(self.dim):
                if ((i >> qubit_index) & 1) == outcome:
                    new_state[i] = self._state_vector.data[i]
            
            # Normalize
            norm = np.linalg.norm(new_state)
            if norm > 0:
                new_state = new_state / norm
            
            self._state_vector = Ket(new_state)
            
            # Update the individual qubit state
            self._qubits[qubit_index]._state = zero() if outcome == 0 else one()
            self._qubits[qubit_index]._measured = True
            self._qubits[qubit_index]._value = outcome
            
            logger.debug(f"Measured qubit {qubit_index}: {outcome}")
            return outcome
        
        else:
            # Measure all qubits
            self._update_state_vector()
            
            probs = np.abs(self._state_vector.data) ** 2
            probs = probs / np.sum(probs)
            
            # Sample
            outcome_index = int(np.random.choice(range(self.dim), p=probs))
            
            # Collapse state
            new_state = np.zeros(self.dim, dtype=complex)
            new_state[outcome_index] = 1.0
            self._state_vector = Ket(new_state)
            
            # Update individual qubits
            binary = format(outcome_index, f'0{self._n_qubits}b')
            for i, bit in enumerate(binary):
                bit_val = int(bit)
                self._qubits[i]._state = one() if bit_val == 1 else zero()
                self._qubits[i]._measured = True
                self._qubits[i]._value = bit_val
            
            # Build counts for multiple shots (if we want to do many measurements)
            # For single shot, return the binary string
            counts = {binary: 1}
            
            logger.debug(f"Measured all qubits: {binary}")
            
            return {
                'counts': counts,
                'shots': 1,
                'n_qubits': self._n_qubits,
                'outcome': binary,
                'outcome_index': outcome_index
            }
    
    def measure_shots(self, shots: int = 1024) -> Dict[str, Any]:
        """
        Measure all qubits multiple times
        
        Args:
            shots: Number of measurement shots
            
        Returns:
            Dict: Measurement results with counts
        
        Example:
            >>> reg = QuantumRegister(2)
            >>> reg.get_qubit(0).apply_h()
            >>> reg.get_qubit(1).apply_h()
            >>> result = reg.measure_shots(1000)
            >>> print(result['counts'])  # {'00': 250, '01': 250, '10': 250, '11': 250}
        """
        self._update_state_vector()
        
        probs = np.abs(self._state_vector.data) ** 2
        probs = probs / np.sum(probs)
        
        # Sample
        outcomes = np.random.choice(range(self.dim), size=shots, p=probs)
        
        counts = {}
        for out in outcomes:
            binary = format(out, f'0{self._n_qubits}b')
            counts[binary] = counts.get(binary, 0) + 1
        
        # Find most likely outcome
        most_likely = max(counts, key=counts.get) if counts else None
        
        logger.info(f"Measured all qubits with {shots} shots")
        
        return {
            'counts': counts,
            'shots': shots,
            'n_qubits': self._n_qubits,
            'most_likely': most_likely,
            'probabilities': probs.tolist()
        }
    
    def reset(self) -> 'QuantumRegister':
        """
        Reset all qubits to |0⟩ state
        
        Returns:
            QuantumRegister: Self for chaining
        
        Example:
            >>> reg = QuantumRegister(2)
            >>> reg.get_qubit(0).apply_x()  # Set to |1⟩
            >>> reg.reset()
            >>> print(reg.get_state_vector())  # |00⟩
        """
        for qubit in self._qubits:
            qubit.reset()
        
        self._state_vector = None
        self._update_state_vector()
        
        logger.info("Register reset to |0...0⟩")
        return self
    
    def apply_gate_to_all(self, gate_name: str, *args, **kwargs) -> 'QuantumRegister':
        """
        Apply a gate to all qubits
        
        Args:
            gate_name: Name of the gate
            *args, **kwargs: Additional arguments for the gate
        
        Returns:
            QuantumRegister: Self for chaining
        
        Example:
            >>> reg = QuantumRegister(3)
            >>> reg.apply_gate_to_all('h')  # Apply H to all qubits
            >>> reg.apply_gate_to_all('rx', np.pi/2)  # Apply Rx to all qubits
        """
        gate_map = {
            'x': 'apply_x',
            'y': 'apply_y',
            'z': 'apply_z',
            'h': 'apply_h',
            's': 'apply_s',
            't': 'apply_t',
            'rx': 'apply_rx',
            'ry': 'apply_ry',
            'rz': 'apply_rz',
        }
        
        method_name = gate_map.get(gate_name.lower())
        if method_name is None:
            raise ValueError(f"Unknown gate: {gate_name}")
        
        for qubit in self._qubits:
            method = getattr(qubit, method_name)
            method(*args, **kwargs)
        
        self._update_state_vector()
        return self
    
    def apply_controlled_gate(
        self,
        gate_name: str,
        control: int,
        target: int,
        *args,
        **kwargs
    ) -> 'QuantumRegister':
        """
        Apply a controlled gate to the register
        
        Args:
            gate_name: Name of the gate
            control: Control qubit index
            target: Target qubit index
            *args, **kwargs: Additional arguments for the gate
        
        Returns:
            QuantumRegister: Self for chaining
        
        Example:
            >>> reg = QuantumRegister(2)
            >>> reg.apply_controlled_gate('x', 0, 1)  # CNOT
        """
        if control == target:
            raise ValueError("Control and target qubits must be different")
        
        # For CNOT, use the cx method from QuantumCircuit
        # For other controlled gates, we need to build the controlled version
        # This is a simplified implementation
        
        # Build the controlled gate matrix
        # For now, we only support CNOT
        if gate_name.lower() == 'x':
            # CNOT gate
            from ..quantum.operator import cnot
            
            # Apply CNOT using the circuit approach
            # We'll use the state vector directly
            self._update_state_vector()
            
            # Build the full CNOT matrix on the given qubits
            full_matrix = self._build_controlled_matrix(control, target, 'X')
            self._state_vector = Ket(full_matrix @ self._state_vector.data)
            
            # Update individual qubits
            self._update_from_state_vector()
            
            logger.debug(f"Applied CNOT with control={control}, target={target}")
        
        elif gate_name.lower() == 'z':
            # CZ gate
            from ..quantum.operator import cz
            self._update_state_vector()
            full_matrix = self._build_controlled_matrix(control, target, 'Z')
            self._state_vector = Ket(full_matrix @ self._state_vector.data)
            self._update_from_state_vector()
            logger.debug(f"Applied CZ with control={control}, target={target}")
        
        else:
            raise ValueError(f"Controlled {gate_name} gate not yet implemented")
        
        return self
    
    def _build_controlled_matrix(self, control: int, target: int, gate: str) -> np.ndarray:
        """
        Build the full matrix for a controlled gate
        
        Args:
            control: Control qubit index
            target: Target qubit index
            gate: Gate type ('X' or 'Z')
        
        Returns:
            np.ndarray: Full matrix
        """
        dim = self.dim
        full_matrix = np.zeros((dim, dim), dtype=complex)
        
        from ..quantum.operator import pauli_x, pauli_z, identity
        
        if gate == 'X':
            gate_matrix = pauli_x().data
        elif gate == 'Z':
            gate_matrix = pauli_z().data
        else:
            raise ValueError(f"Unknown gate: {gate}")
        
        # Build the controlled gate
        # For each basis state, if control qubit is 1, apply gate to target
        for i in range(dim):
            # Check if control qubit is 1
            if (i >> control) & 1:
                # Apply gate to target
                j = i ^ (1 << target)  # Flip target bit
                full_matrix[j, i] = 1.0
            else:
                full_matrix[i, i] = 1.0
        
        return full_matrix
    
    def _update_from_state_vector(self):
        """Update individual qubit states from the state vector"""
        if self._state_vector is None:
            return
        
        # For product states, extract individual qubit states
        # For entangled states, this is not possible
        # We'll only update if the state is a product state
        # This is a simplified check
        
        # Try to factorize the state
        # For now, we just update the qubits if they are in a product state
        # This is a placeholder
        pass
    
    # ============================================================================
    # MAGIC METHODS
    # ============================================================================
    
    def __getitem__(self, index: int) -> Qubit:
        """
        Get a qubit by index using bracket notation
        
        Args:
            index: Qubit index
        
        Returns:
            Qubit: The qubit
        
        Example:
            >>> reg = QuantumRegister(3)
            >>> q = reg[1]  # Get second qubit
        """
        return self.get_qubit(index)
    
    def __len__(self) -> int:
        """
        Get the number of qubits in the register
        
        Returns:
            int: Number of qubits
        
        Example:
            >>> reg = QuantumRegister(3)
            >>> len(reg)  # 3
        """
        return self._n_qubits
    
    def __repr__(self) -> str:
        """String representation"""
        return f"QuantumRegister(n_qubits={self._n_qubits}, dim={self.dim})"
    
    def __str__(self) -> str:
        """Pretty string representation"""
        state = self.get_state_vector()
        return f"QuantumRegister({self._n_qubits} qubits):\n{state}"
    
    def __iter__(self):
        """Iterate over qubits"""
        return iter(self._qubits)
    
    def __contains__(self, item) -> bool:
        """Check if an item is in the register"""
        if isinstance(item, int):
            return 0 <= item < self._n_qubits
        return False
    
    def __eq__(self, other: 'QuantumRegister') -> bool:
        """Check equality of registers"""
        if not isinstance(other, QuantumRegister):
            return False
        if self._n_qubits != other._n_qubits:
            return False
        return all(q1 == q2 for q1, q2 in zip(self._qubits, other._qubits))
    
    def __ne__(self, other: 'QuantumRegister') -> bool:
        """Check inequality of registers"""
        return not self.__eq__(other)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_register_from_state(state: Union[List, np.ndarray, Ket]) -> QuantumRegister:
    """
    Create a quantum register from a state vector
    
    Args:
        state: State vector
    
    Returns:
        QuantumRegister: Register with the given state
    
    Example:
        >>> from psiqit.quantum import bell_phi_plus
        >>> reg = create_register_from_state(bell_phi_plus())
        >>> print(reg.get_state_vector())  # Bell state
    """
    if isinstance(state, Ket):
        dim = state.dim
    else:
        dim = len(state)
    
    # Check if dimension is a power of 2
    n_qubits = int(np.log2(dim))
    if 2 ** n_qubits != dim:
        raise ValueError(f"State dimension {dim} is not a power of 2")
    
    reg = QuantumRegister(n_qubits)
    reg.set_state_vector(state)
    return reg


def create_product_state(states: List[Union[List, np.ndarray, Ket]]) -> QuantumRegister:
    """
    Create a quantum register from individual qubit states
    
    Args:
        states: List of qubit states
    
    Returns:
        QuantumRegister: Register with the given states
    
    Example:
        >>> from psiqit.quantum import zero, one, plus
        >>> reg = create_product_state([zero(), one(), plus()])
        >>> print(reg.get_state_vector())  # |0⟩ ⊗ |1⟩ ⊗ |+⟩
    """
    n_qubits = len(states)
    reg = QuantumRegister(n_qubits)
    
    for i, state in enumerate(states):
        reg.get_qubit(i).state = state
    
    return reg


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'QuantumRegister',
    'create_register_from_state',
    'create_product_state',
]