#psiqit/circuits/optical_circuits.py

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from ..quantum.state import Ket, basis
from ..quantum.operator import Operator, identity, rx, ry, rz, hadamard, phase
from ..utils.logger import logger
from ..utils.validation import validate_qubits
from .circuit import QuantumCircuit


class BeamSplitter:
    """
    Optical beam splitter
    
    A beam splitter is a two-mode optical element that splits light
    into two paths with a given transmittance.
    
    Example:
        >>> bs = BeamSplitter(transmittance=0.5)  # 50:50 beam splitter
        >>> state = Ket([1, 0])  # Input in mode 0
        >>> output = bs.apply(state)
        >>> print(output)  # (|0⟩ + i|1⟩)/√2
    """
    
    def __init__(self, transmittance: float = 0.5):
        """
        Initialize a beam splitter
        
        Args:
            transmittance: Transmittance (0 to 1), 0.5 is 50:50
        
        Example:
            >>> bs = BeamSplitter(0.5)  # 50:50
            >>> bs = BeamSplitter(0.7)  # 70% transmission, 30% reflection
        """
        if not 0 <= transmittance <= 1:
            raise ValueError(f"Transmittance must be between 0 and 1, got {transmittance}")
        
        self._transmittance = transmittance
        self._matrix = self._compute_matrix()
        
        logger.debug(f"BeamSplitter initialized with T={transmittance:.3f}")
    
    def _compute_matrix(self) -> np.ndarray:
        """
        Compute the beam splitter matrix
        
        Returns:
            np.ndarray: 2x2 unitary matrix
        
        The standard beam splitter matrix:
        [[√T,  i√R],
         [i√R,  √T]]
        where T is transmittance and R = 1 - T is reflectance
        """
        T = self._transmittance
        R = 1 - T
        sqrt_T = np.sqrt(T)
        sqrt_R = np.sqrt(R)
        
        return np.array([
            [sqrt_T, 1j * sqrt_R],
            [1j * sqrt_R, sqrt_T]
        ], dtype=complex)
    
    @property
    def matrix(self) -> np.ndarray:
        """Get the beam splitter matrix"""
        return self._matrix.copy()
    
    @property
    def transmittance(self) -> float:
        """Get the transmittance"""
        return self._transmittance
    
    @property
    def reflectance(self) -> float:
        """Get the reflectance (1 - transmittance)"""
        return 1 - self._transmittance
    
    def apply(self, state: Union[List, np.ndarray, Ket]) -> Ket:
        """
        Apply the beam splitter to a state
        
        Args:
            state: Input state (2-dimensional)
            
        Returns:
            Ket: Output state
        
        Example:
            >>> bs = BeamSplitter(0.5)
            >>> state = Ket([1, 0])  # |0⟩
            >>> output = bs.apply(state)
            >>> print(output)  # (|0⟩ + i|1⟩)/√2
        """
        if isinstance(state, Ket):
            state_data = state.data
        else:
            state_data = np.array(state, dtype=complex)
        
        if len(state_data) != 2:
            raise ValueError(f"State must be 2-dimensional, got {len(state_data)}")
        
        output = self._matrix @ state_data
        return Ket(output)
    
    def as_gate(self) -> Operator:
        """
        Convert the beam splitter to a quantum gate
        
        Returns:
            Operator: Beam splitter operator
        
        Example:
            >>> bs = BeamSplitter(0.5)
            >>> U = bs.as_gate()
            >>> print(U.is_unitary)  # True
        """
        return Operator(self._matrix, name=f"BS(T={self._transmittance:.2f})")
    
    def as_circuit(self, q0: int, q1: int) -> QuantumCircuit:
        """
        Create a quantum circuit representation of the beam splitter
        
        Args:
            q0: First mode/qubit
            q1: Second mode/qubit
            
        Returns:
            QuantumCircuit: Circuit implementing the beam splitter
        
        Note:
            This uses a decomposition of the beam splitter into
            rotation gates for quantum circuit implementation.
        """
        # Beam splitter can be decomposed as:
        # BS(θ) = Ry(θ) ⊗ I  with appropriate rotations
        # For 50:50, it's equivalent to Hadamard on the two modes
        
        n_qubits = max(q0, q1) + 1
        circ = QuantumCircuit(n_qubits)
        
        if abs(self._transmittance - 0.5) < 1e-10:
            # 50:50 beam splitter is equivalent to Hadamard
            circ.h(q0)
            circ.h(q1)
            # Add a phase shift on q1
            circ.rz(q1, np.pi/2)
        else:
            # General beam splitter decomposition
            # BS(θ) = Ry(θ) ⊗ I with θ = arccos(√T)
            theta = 2 * np.arccos(np.sqrt(self._transmittance))
            circ.ry(q0, theta)
            circ.ry(q1, theta)
            # Add phase shift
            circ.rz(q1, np.pi/2)
        
        logger.debug(f"Beam splitter circuit created on qubits {q0}, {q1}")
        return circ
    
    def __repr__(self) -> str:
        """String representation"""
        return f"BeamSplitter(T={self._transmittance:.3f}, R={self.reflectance:.3f})"
    
    def __str__(self) -> str:
        """Pretty string representation"""
        return f"BS(T={self._transmittance:.2f})"


class PhaseShifter:
    """
    Optical phase shifter
    
    A phase shifter introduces a phase shift to a mode.
    
    Example:
        >>> ps = PhaseShifter(phase=np.pi/2)  # 90° phase shift
        >>> state = Ket([1, 0])
        >>> output = ps.apply(state)
        >>> print(output)  # |0⟩ with phase e^{iπ/2}
    """
    
    def __init__(self, phase: float = 0.0):
        """
        Initialize a phase shifter
        
        Args:
            phase: Phase shift in radians
        
        Example:
            >>> ps = PhaseShifter(0)  # No phase shift
            >>> ps = PhaseShifter(np.pi)  # π phase shift
            >>> ps = PhaseShifter(np.pi/2)  # π/2 phase shift
        """
        self._phase = phase
        self._matrix = self._compute_matrix()
        
        logger.debug(f"PhaseShifter initialized with phase={phase:.3f}")
    
    def _compute_matrix(self) -> np.ndarray:
        """
        Compute the phase shifter matrix
        
        Returns:
            np.ndarray: 2x2 diagonal matrix
        
        The phase shifter matrix:
        [[1, 0],
         [0, e^{iφ}]]
        """
        return np.array([
            [1, 0],
            [0, np.exp(1j * self._phase)]
        ], dtype=complex)
    
    @property
    def phase(self) -> float:
        """Get the current phase shift"""
        return self._phase
    
    @property
    def matrix(self) -> np.ndarray:
        """Get the phase shifter matrix"""
        return self._matrix.copy()
    
    def set_phase(self, phase: float) -> 'PhaseShifter':
        """
        Set the phase shift
        
        Args:
            phase: New phase shift in radians
            
        Returns:
            PhaseShifter: Self for chaining
        
        Example:
            >>> ps = PhaseShifter()
            >>> ps.set_phase(np.pi/2)
            >>> print(ps.phase)  # 1.5708
        """
        self._phase = phase
        self._matrix = self._compute_matrix()
        logger.debug(f"Phase set to {phase:.3f}")
        return self
    
    def apply(self, state: Union[List, np.ndarray, Ket]) -> Ket:
        """
        Apply the phase shifter to a state
        
        Args:
            state: Input state (2-dimensional)
            
        Returns:
            Ket: Output state
        
        Example:
            >>> ps = PhaseShifter(np.pi)
            >>> state = Ket([0, 1])  # |1⟩
            >>> output = ps.apply(state)
            >>> print(output)  # -|1⟩
        """
        if isinstance(state, Ket):
            state_data = state.data
        else:
            state_data = np.array(state, dtype=complex)
        
        if len(state_data) != 2:
            raise ValueError(f"State must be 2-dimensional, got {len(state_data)}")
        
        output = self._matrix @ state_data
        return Ket(output)
    
    def as_gate(self) -> Operator:
        """
        Convert the phase shifter to a quantum gate
        
        Returns:
            Operator: Phase shifter operator
        
        Example:
            >>> ps = PhaseShifter(np.pi/2)
            >>> U = ps.as_gate()
            >>> print(U.is_unitary)  # True
        """
        return Operator(self._matrix, name=f"P(φ={self._phase:.2f})")
    
    def as_circuit(self, qubit: int) -> QuantumCircuit:
        """
        Create a quantum circuit representation of the phase shifter
        
        Args:
            qubit: Qubit/mode to apply the phase shift to
            
        Returns:
            QuantumCircuit: Circuit implementing the phase shifter
        
        Example:
            >>> ps = PhaseShifter(np.pi/2)
            >>> circ = ps.as_circuit(0)
            >>> print(circ.draw())
        """
        n_qubits = qubit + 1
        circ = QuantumCircuit(n_qubits)
        circ.rz(qubit, self._phase)
        return circ
    
    def __repr__(self) -> str:
        """String representation"""
        return f"PhaseShifter(phase={self._phase:.3f})"
    
    def __str__(self) -> str:
        """Pretty string representation"""
        return f"PS(φ={self._phase:.2f})"


class OpticalCircuit:
    """
    Optical quantum circuit
    
    A circuit composed of beam splitters and phase shifters.
    
    Example:
        >>> oc = OpticalCircuit(2)
        >>> oc.add_beam_splitter(0, 1, 0.5)  # 50:50 beam splitter
        >>> oc.add_phase_shifter(0, np.pi/2)  # Phase shift on mode 0
        >>> state = oc.simulate(Ket([1, 0]))
        >>> print(state)
    """
    
    def __init__(self, n_modes: int):
        """
        Initialize an optical circuit
        
        Args:
            n_modes: Number of optical modes
        
        Example:
            >>> oc = OpticalCircuit(2)  # 2-mode circuit
            >>> oc = OpticalCircuit(4)  # 4-mode circuit
        """
        validate_qubits(n_modes)
        self._n_modes = n_modes
        self._elements: List[Dict[str, Any]] = []
        self._total_matrix = None
        
        logger.info(f"OpticalCircuit initialized with {n_modes} modes")
    
    @property
    def n_modes(self) -> int:
        """Get the number of modes"""
        return self._n_modes
    
    def add_beam_splitter(
        self,
        mode1: int,
        mode2: int,
        transmittance: float = 0.5
    ) -> 'OpticalCircuit':
        """
        Add a beam splitter between two modes
        
        Args:
            mode1: First mode
            mode2: Second mode
            transmittance: Transmittance (0 to 1)
            
        Returns:
            OpticalCircuit: Self for chaining
        
        Example:
            >>> oc = OpticalCircuit(2)
            >>> oc.add_beam_splitter(0, 1, 0.5)
        """
        if mode1 == mode2:
            raise ValueError("Modes must be different")
        if mode1 < 0 or mode1 >= self._n_modes or mode2 < 0 or mode2 >= self._n_modes:
            raise ValueError(f"Modes must be between 0 and {self._n_modes-1}")
        
        self._elements.append({
            'type': 'beam_splitter',
            'mode1': mode1,
            'mode2': mode2,
            'transmittance': transmittance
        })
        
        self._total_matrix = None  # Invalidate cached matrix
        logger.debug(f"Added beam splitter between {mode1} and {mode2}")
        return self
    
    def add_phase_shifter(self, mode: int, phase: float) -> 'OpticalCircuit':
        """
        Add a phase shifter to a mode
        
        Args:
            mode: Mode to apply phase shift to
            phase: Phase shift in radians
            
        Returns:
            OpticalCircuit: Self for chaining
        
        Example:
            >>> oc = OpticalCircuit(2)
            >>> oc.add_phase_shifter(0, np.pi/2)
        """
        if mode < 0 or mode >= self._n_modes:
            raise ValueError(f"Mode must be between 0 and {self._n_modes-1}")
        
        self._elements.append({
            'type': 'phase_shifter',
            'mode': mode,
            'phase': phase
        })
        
        self._total_matrix = None  # Invalidate cached matrix
        logger.debug(f"Added phase shifter on mode {mode} with phase {phase:.3f}")
        return self
    
    def build_circuit(self) -> QuantumCircuit:
        """
        Build a quantum circuit from the optical elements
        
        Returns:
            QuantumCircuit: Quantum circuit representation
        
        Example:
            >>> oc = OpticalCircuit(2)
            >>> oc.add_beam_splitter(0, 1, 0.5)
            >>> circ = oc.build_circuit()
            >>> print(circ.draw())
        """
        n_qubits = self._n_modes
        circ = QuantumCircuit(n_qubits)
        
        for element in self._elements:
            if element['type'] == 'beam_splitter':
                mode1 = element['mode1']
                mode2 = element['mode2']
                T = element['transmittance']
                
                # Beam splitter decomposition
                if abs(T - 0.5) < 1e-10:
                    # 50:50 beam splitter
                    circ.h(mode1)
                    circ.h(mode2)
                    circ.rz(mode2, np.pi/2)
                else:
                    theta = 2 * np.arccos(np.sqrt(T))
                    circ.ry(mode1, theta)
                    circ.ry(mode2, theta)
                    circ.rz(mode2, np.pi/2)
            
            elif element['type'] == 'phase_shifter':
                mode = element['mode']
                phase = element['phase']
                circ.rz(mode, phase)
        
        logger.debug(f"Circuit built with {len(self._elements)} elements")
        return circ
    
    def get_matrix(self) -> np.ndarray:
        """
        Get the total unitary matrix of the circuit
        
        Returns:
            np.ndarray: Total unitary matrix
        
        Example:
            >>> oc = OpticalCircuit(2)
            >>> oc.add_beam_splitter(0, 1, 0.5)
            >>> U = oc.get_matrix()
            >>> print(U.shape)  # (4, 4)
        """
        if self._total_matrix is not None:
            return self._total_matrix
        
        # Build the total matrix from the elements
        # Start with identity
        total = np.eye(2 ** self._n_modes, dtype=complex)
        
        for element in self._elements:
            if element['type'] == 'beam_splitter':
                mode1 = element['mode1']
                mode2 = element['mode2']
                T = element['transmittance']
                
                # Beam splitter matrix
                bs = BeamSplitter(T)
                bs_matrix = bs.matrix
                
                # Expand to full Hilbert space
                full_matrix = self._expand_2mode_matrix(bs_matrix, mode1, mode2)
                total = full_matrix @ total
            
            elif element['type'] == 'phase_shifter':
                mode = element['mode']
                phase = element['phase']
                
                # Phase shifter matrix
                ps = PhaseShifter(phase)
                ps_matrix = ps.matrix
                
                # Expand to full Hilbert space
                full_matrix = self._expand_1mode_matrix(ps_matrix, mode)
                total = full_matrix @ total
        
        self._total_matrix = total
        return total
    
    def _expand_1mode_matrix(self, matrix: np.ndarray, mode: int) -> np.ndarray:
        """
        Expand a 2x2 matrix to act on the full Hilbert space
        
        Args:
            matrix: 2x2 matrix
            mode: Mode to apply to
            
        Returns:
            np.ndarray: Full matrix
        """
        dim = 2 ** self._n_modes
        full = np.eye(dim, dtype=complex)
        
        # Build the full matrix as tensor product
        # This is a simplified version
        for i in range(self._n_modes):
            if i == mode:
                full = np.kron(full, matrix)
            else:
                full = np.kron(full, np.eye(2, dtype=complex))
        
        return full
    
    def _expand_2mode_matrix(self, matrix: np.ndarray, mode1: int, mode2: int) -> np.ndarray:
        """
        Expand a 4x4 matrix to act on the full Hilbert space
        
        Args:
            matrix: 4x4 matrix
            mode1: First mode
            mode2: Second mode
            
        Returns:
            np.ndarray: Full matrix
        """
        dim = 2 ** self._n_modes
        full = np.eye(dim, dtype=complex)
        
        # For now, we use a simplified approach
        # For 2 modes, the matrix is directly applied
        if self._n_modes == 2:
            return matrix
        
        # For more than 2 modes, we need to expand
        # This is a placeholder for full expansion
        # In practice, we would use permutation matrices
        logger.warning(f"Expanding 2-mode matrix for {self._n_modes} modes using simplified method")
        
        # Build full matrix
        for i in range(self._n_modes):
            if i == mode1:
                # Apply matrix to modes mode1 and mode2
                full = np.kron(full, matrix)
            elif i == mode2:
                continue  # Already included
            else:
                full = np.kron(full, np.eye(2, dtype=complex))
        
        return full
    
    def simulate(self, input_state: Optional[Union[List, np.ndarray, Ket]] = None) -> Ket:
        """
        Simulate the optical circuit
        
        Args:
            input_state: Input state (default: |0...0⟩)
            
        Returns:
            Ket: Output state
        
        Example:
            >>> oc = OpticalCircuit(2)
            >>> oc.add_beam_splitter(0, 1, 0.5)
            >>> state = oc.simulate(Ket([1, 0, 0, 0]))  # Input |00⟩
            >>> print(state)  # Output state
        """
        if input_state is None:
            # Default to |0...0⟩
            dim = 2 ** self._n_modes
            state = Ket(np.zeros(dim, dtype=complex))
            state.data[0] = 1.0
        elif isinstance(input_state, Ket):
            state = input_state.copy()
        else:
            state = Ket(np.array(input_state, dtype=complex))
        
        if state.dim != 2 ** self._n_modes:
            raise ValueError(f"State dimension {state.dim} does not match circuit dimension {2 ** self._n_modes}")
        
        # Apply the total matrix
        U = self.get_matrix()
        output = U @ state.data
        
        logger.info(f"Optical circuit simulated with {len(self._elements)} elements")
        return Ket(output)
    
    def clear(self) -> 'OpticalCircuit':
        """
        Clear all elements from the circuit
        
        Returns:
            OpticalCircuit: Self for chaining
        
        Example:
            >>> oc = OpticalCircuit(2)
            >>> oc.add_beam_splitter(0, 1, 0.5)
            >>> oc.clear()
            >>> len(oc)  # 0
        """
        self._elements = []
        self._total_matrix = None
        logger.info("Circuit cleared")
        return self
    
    # ============================================================================
    # MAGIC METHODS
    # ============================================================================
    
    def __len__(self) -> int:
        """Get the number of elements in the circuit"""
        return len(self._elements)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"OpticalCircuit(n_modes={self._n_modes}, elements={len(self._elements)})"
    
    def __str__(self) -> str:
        """Pretty string representation"""
        lines = [f"OpticalCircuit with {self._n_modes} modes, {len(self._elements)} elements:"]
        for i, elem in enumerate(self._elements):
            if elem['type'] == 'beam_splitter':
                lines.append(f"  {i}: BS({elem['mode1']}, {elem['mode2']}, T={elem['transmittance']:.3f})")
            elif elem['type'] == 'phase_shifter':
                lines.append(f"  {i}: PS({elem['mode']}, φ={elem['phase']:.3f})")
        return "\n".join(lines)
    
    def __iter__(self):
        """Iterate over elements"""
        return iter(self._elements)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get an element by index"""
        return self._elements[index]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def beam_splitter_circuit(
    q0: int,
    q1: int,
    transmittance: float = 0.5
) -> QuantumCircuit:
    """
    Create a circuit with a beam splitter
    
    Args:
        q0: First qubit
        q1: Second qubit
        transmittance: Transmittance (0 to 1)
        
    Returns:
        QuantumCircuit: Circuit with beam splitter
    
    Example:
        >>> circ = beam_splitter_circuit(0, 1, 0.5)
        >>> print(circ.draw())
    """
    bs = BeamSplitter(transmittance)
    return bs.as_circuit(q0, q1)


def phase_shifter_circuit(
    qubit: int,
    phase: float
) -> QuantumCircuit:
    """
    Create a circuit with a phase shifter
    
    Args:
        qubit: Qubit to apply phase shift to
        phase: Phase shift in radians
        
    Returns:
        QuantumCircuit: Circuit with phase shifter
    
    Example:
        >>> circ = phase_shifter_circuit(0, np.pi/2)
        >>> print(circ.draw())
    """
    ps = PhaseShifter(phase)
    return ps.as_circuit(qubit)


def mach_zehnder_interferometer(
    q0: int,
    q1: int,
    phase: float = 0.0
) -> QuantumCircuit:
    """
    Create a Mach-Zehnder interferometer circuit
    
    Args:
        q0: First qubit
        q1: Second qubit
        phase: Phase shift in one arm
        
    Returns:
        QuantumCircuit: Mach-Zehnder interferometer
    
    Example:
        >>> circ = mach_zehnder_interferometer(0, 1, np.pi/2)
        >>> print(circ.draw())
    """
    n_qubits = max(q0, q1) + 1
    circ = QuantumCircuit(n_qubits)
    
    # First beam splitter (50:50)
    circ.h(q0)
    circ.h(q1)
    circ.rz(q1, np.pi/2)
    
    # Phase shifter in one arm
    circ.rz(q0, phase)
    
    # Second beam splitter (50:50)
    circ.h(q0)
    circ.h(q1)
    circ.rz(q1, np.pi/2)
    
    return circ


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'BeamSplitter',
    'PhaseShifter',
    'OpticalCircuit',
    'beam_splitter_circuit',
    'phase_shifter_circuit',
    'mach_zehnder_interferometer',
]