#psiqit/quantum/parties.py


import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from ..math.qalgebra import PI, SQRT2
from ..quantum.state import Ket, zero, one, plus, minus, bell_phi_plus, bell_phi_minus, bell_psi_plus, bell_psi_minus, basis
from ..quantum.operator import pauli_x, pauli_z, hadamard, cnot, identity, rx, ry, rz
from ..utils.logger import logger
from ..utils.random import set_random_seed, random_state


# ============================================================================
# ALICE CLASS
# ============================================================================

class Alice:
    """
    Alice - Quantum communication party
    
    Alice can prepare quantum states, encode bits, and send states to Bob.
    
    Example:
        >>> alice = Alice()
        >>> state = alice.prepare(plus()).send()
        >>> print(state)
    """
    
    def __init__(self, name: str = "Alice"):
        """
        Initialize Alice
        
        Args:
            name: Name of the party
        """
        self.name = name
        self._state = None
        self._encoded_bits = []
        self._measurements = []
        
        logger.info(f"{self.name} initialized")
    
    def prepare(self, state: Ket) -> 'Alice':
        """
        Prepare a quantum state
        
        Args:
            state: Quantum state to prepare
            
        Returns:
            Alice: Self for chaining
        """
        self._state = state.copy()
        logger.debug(f"{self.name} prepared state: {state}")
        return self
    
    def prepare_random(self, n_qubits: int = 1) -> Ket:
        """
        Prepare a random quantum state
        
        Args:
            n_qubits: Number of qubits
            
        Returns:
            Ket: Random state
            
        Example:
            >>> alice = Alice()
            >>> state = alice.prepare_random(2)
            >>> print(state.dim)  # 4
        """
        self._state = random_state(2 ** n_qubits)
        logger.debug(f"{self.name} prepared random {n_qubits}-qubit state")
        return self._state
    
    def send(self) -> Ket:
        """
        Send the prepared state
        
        Returns:
            Ket: The state to send
        """
        if self._state is None:
            raise ValueError(f"{self.name} has no state to send. Call prepare() first.")
        
        logger.info(f"{self.name} sent a state")
        return self._state.copy()
    
    def measure(self, basis: str = 'z') -> int:
        """
        Measure the current state
        
        Args:
            basis: 'z', 'x', or 'y'
            
        Returns:
            int: Measurement outcome (0 or 1)
            
        Example:
            >>> alice = Alice()
            >>> alice.prepare(zero())
            >>> result = alice.measure('z')  # 0
        """
        if self._state is None:
            raise ValueError(f"{self.name} has no state to measure")
        
        # Choose measurement basis
        if basis == 'z':
            # Project onto |0⟩ and |1⟩
            probs = [abs(self._state.data[0])**2, abs(self._state.data[1])**2]
        elif basis == 'x':
            # Project onto |+⟩ and |-⟩
            plus_state = plus()
            minus_state = minus()
            probs = [
                abs(np.vdot(plus_state.data, self._state.data))**2,
                abs(np.vdot(minus_state.data, self._state.data))**2
            ]
        elif basis == 'y':
            # Project onto |i⟩ and |-i⟩
            i_state = Ket([1/np.sqrt(2), 1j/np.sqrt(2)])
            minus_i_state = Ket([1/np.sqrt(2), -1j/np.sqrt(2)])
            probs = [
                abs(np.vdot(i_state.data, self._state.data))**2,
                abs(np.vdot(minus_i_state.data, self._state.data))**2
            ]
        else:
            raise ValueError(f"Unknown basis: {basis}. Use 'z', 'x', or 'y'")
        
        # Normalize probabilities
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        
        # Sample
        outcome = int(np.random.choice([0, 1], p=probs))
        self._measurements.append(outcome)
        
        logger.debug(f"{self.name} measured {outcome} in {basis}-basis")
        return outcome
    
    def encode_bit(self, bit: int) -> 'Alice':
        """
        Encode a classical bit into the current state
        
        For single qubit:
        - bit 0: apply I (no change)
        - bit 1: apply X (flip)
        
        Args:
            bit: Bit to encode (0 or 1)
            
        Returns:
            Alice: Self for chaining
        """
        if self._state is None:
            raise ValueError(f"{self.name} has no state to encode")
        
        if bit not in [0, 1]:
            raise ValueError(f"Bit must be 0 or 1, got {bit}")
        
        if bit == 1:
            # Apply X gate to flip
            X = pauli_x()
            self._state = Ket(X.data @ self._state.data)
        
        self._encoded_bits.append(bit)
        logger.debug(f"{self.name} encoded bit {bit}")
        return self
    
    def get_measurements(self) -> List[int]:
        """Get all measurements performed"""
        return self._measurements
    
    def get_encoded_bits(self) -> List[int]:
        """Get all encoded bits"""
        return self._encoded_bits
    
    def reset(self) -> 'Alice':
        """Reset Alice's state"""
        self._state = None
        self._encoded_bits = []
        self._measurements = []
        return self
    
    def __repr__(self) -> str:
        return f"Alice(state={'prepared' if self._state is not None else 'empty'})"


# ============================================================================
# BOB CLASS
# ============================================================================

class Bob:
    """
    Bob - Quantum communication party
    
    Bob can receive states and measure them.
    
    Example:
        >>> bob = Bob()
        >>> state = bob.receive(alice.send())
        >>> result = bob.measure('x')
    """
    
    def __init__(self, name: str = "Bob"):
        """
        Initialize Bob
        
        Args:
            name: Name of the party
        """
        self.name = name
        self._state = None
        self._measurements = []
        self._received_bits = []
        
        logger.info(f"{self.name} initialized")
    
    def receive(self, state: Ket) -> 'Bob':
        """
        Receive a quantum state
        
        Args:
            state: State to receive
            
        Returns:
            Bob: Self for chaining
        """
        self._state = state.copy()
        logger.debug(f"{self.name} received state")
        return self
    
    def measure(self, basis: str = 'z') -> int:
        """
        Measure the received state
        
        Args:
            basis: 'z', 'x', or 'y'
            
        Returns:
            int: Measurement outcome (0 or 1)
        """
        if self._state is None:
            raise ValueError(f"{self.name} has no state to measure")
        
        # Choose measurement basis
        if basis == 'z':
            probs = [abs(self._state.data[0])**2, abs(self._state.data[1])**2]
        elif basis == 'x':
            plus_state = plus()
            minus_state = minus()
            probs = [
                abs(np.vdot(plus_state.data, self._state.data))**2,
                abs(np.vdot(minus_state.data, self._state.data))**2
            ]
        elif basis == 'y':
            i_state = Ket([1/np.sqrt(2), 1j/np.sqrt(2)])
            minus_i_state = Ket([1/np.sqrt(2), -1j/np.sqrt(2)])
            probs = [
                abs(np.vdot(i_state.data, self._state.data))**2,
                abs(np.vdot(minus_i_state.data, self._state.data))**2
            ]
        else:
            raise ValueError(f"Unknown basis: {basis}. Use 'z', 'x', or 'y'")
        
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        
        outcome = int(np.random.choice([0, 1], p=probs))
        self._measurements.append(outcome)
        
        logger.debug(f"{self.name} measured {outcome} in {basis}-basis")
        return outcome
    
    def get_bit(self) -> int:
        """
        Get the last measured bit
        
        Returns:
            int: Last measurement outcome
        """
        if not self._measurements:
            raise ValueError(f"{self.name} has no measurements")
        return self._measurements[-1]
    
    def get_measurements(self) -> List[int]:
        """Get all measurements performed"""
        return self._measurements
    
    def reset(self) -> 'Bob':
        """Reset Bob's state"""
        self._state = None
        self._measurements = []
        return self
    
    def __repr__(self) -> str:
        return f"Bob(state={'received' if self._state is not None else 'empty'})"


# ============================================================================
# CHARLIE CLASS
# ============================================================================

class Charlie:
    """
    Charlie - Third party for quantum protocols
    
    Charlie can receive states and measure them.
    
    Example:
        >>> charlie = Charlie()
        >>> state = charlie.receive(alice.send())
        >>> result = charlie.measure()
    """
    
    def __init__(self, name: str = "Charlie"):
        """
        Initialize Charlie
        
        Args:
            name: Name of the party
        """
        self.name = name
        self._state = None
        self._measurements = []
        
        logger.info(f"{self.name} initialized")
    
    def receive(self, state: Ket) -> 'Charlie':
        """
        Receive a quantum state
        
        Args:
            state: State to receive
            
        Returns:
            Charlie: Self for chaining
        """
        self._state = state.copy()
        logger.debug(f"{self.name} received state")
        return self
    
    def measure(self, basis: str = 'z') -> int:
        """
        Measure the received state
        
        Args:
            basis: 'z', 'x', or 'y'
            
        Returns:
            int: Measurement outcome (0 or 1)
        """
        if self._state is None:
            raise ValueError(f"{self.name} has no state to measure")
        
        if basis == 'z':
            probs = [abs(self._state.data[0])**2, abs(self._state.data[1])**2]
        elif basis == 'x':
            plus_state = plus()
            minus_state = minus()
            probs = [
                abs(np.vdot(plus_state.data, self._state.data))**2,
                abs(np.vdot(minus_state.data, self._state.data))**2
            ]
        elif basis == 'y':
            i_state = Ket([1/np.sqrt(2), 1j/np.sqrt(2)])
            minus_i_state = Ket([1/np.sqrt(2), -1j/np.sqrt(2)])
            probs = [
                abs(np.vdot(i_state.data, self._state.data))**2,
                abs(np.vdot(minus_i_state.data, self._state.data))**2
            ]
        else:
            raise ValueError(f"Unknown basis: {basis}")
        
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        
        outcome = int(np.random.choice([0, 1], p=probs))
        self._measurements.append(outcome)
        
        logger.debug(f"{self.name} measured {outcome}")
        return outcome
    
    def get_measurements(self) -> List[int]:
        """Get all measurements performed"""
        return self._measurements
    
    def reset(self) -> 'Charlie':
        """Reset Charlie's state"""
        self._state = None
        self._measurements = []
        return self
    
    def __repr__(self) -> str:
        return f"Charlie(state={'received' if self._state is not None else 'empty'})"


# ============================================================================
# BB84 PROTOCOL
# ============================================================================

class BB84:
    """
    BB84 Quantum Key Distribution protocol
    
    The BB84 protocol allows two parties (Alice and Bob) to establish
    a shared secret key using quantum states.
    
    Example:
        >>> bb84 = BB84()
        >>> key = bb84.run(n_bits=100)
        >>> print(f"Key: {key[:20]}...")
    """
    
    def __init__(self, eavesdropping: bool = False):
        """
        Initialize BB84 protocol
        
        Args:
            eavesdropping: If True, simulate Eve (eavesdropper)
        """
        self.alice = Alice()
        self.bob = Bob()
        self.eve = Alice("Eve") if eavesdropping else None
        self.eavesdropping = eavesdropping
        
        self._key = ""
        self._raw_key = []
        self._bases = []
        
        logger.info(f"BB84 initialized (eavesdropping={eavesdropping})")
    
    def run(self, n_bits: int = 100) -> str:
        """
        Run the BB84 protocol
        
        Args:
            n_bits: Number of bits to exchange
            
        Returns:
            str: Shared secret key
            
        Example:
            >>> bb84 = BB84()
            >>> key = bb84.run(50)
            >>> print(len(key))  # ~25 (after sifting)
        """
        logger.info(f"Running BB84 with {n_bits} bits")
        
        # Alice's preparation
        alice_bits = np.random.choice([0, 1], size=n_bits)
        alice_bases = np.random.choice(['z', 'x'], size=n_bits)
        
        # Bob's measurement bases
        bob_bases = np.random.choice(['z', 'x'], size=n_bits)
        
        # Store for later
        self._alice_bits = alice_bits
        self._alice_bases = alice_bases
        self._bob_bases = bob_bases
        
        # Simulate transmission
        bob_bits = []
        
        for i in range(n_bits):
            # Alice prepares state
            bit = alice_bits[i]
            basis = alice_bases[i]
            
            if basis == 'z':
                state = zero() if bit == 0 else one()
            else:  # 'x'
                state = plus() if bit == 0 else minus()
            
            # Send through channel
            sent_state = state.copy()
            
            # Eavesdropping (if enabled)
            if self.eavesdropping and np.random.random() < 0.5:
                # Eve intercepts and measures
                eve_basis = np.random.choice(['z', 'x'])
                if eve_basis == 'z':
                    probs = [abs(sent_state.data[0])**2, abs(sent_state.data[1])**2]
                    outcome = int(np.random.choice([0, 1], p=probs/np.sum(probs)))
                    sent_state = zero() if outcome == 0 else one()
                else:
                    plus_state = plus()
                    minus_state = minus()
                    probs = [
                        abs(np.vdot(plus_state.data, sent_state.data))**2,
                        abs(np.vdot(minus_state.data, sent_state.data))**2
                    ]
                    outcome = int(np.random.choice([0, 1], p=probs/np.sum(probs)))
                    sent_state = plus() if outcome == 0 else minus()
            
            # Bob measures
            bob_basis = bob_bases[i]
            
            if bob_basis == 'z':
                probs = [abs(sent_state.data[0])**2, abs(sent_state.data[1])**2]
                bob_bit = int(np.random.choice([0, 1], p=probs/np.sum(probs)))
            else:  # 'x'
                plus_state = plus()
                minus_state = minus()
                probs = [
                    abs(np.vdot(plus_state.data, sent_state.data))**2,
                    abs(np.vdot(minus_state.data, sent_state.data))**2
                ]
                bob_bit = int(np.random.choice([0, 1], p=probs/np.sum(probs)))
            
            bob_bits.append(bob_bit)
        
        # Sifting: keep only when bases match
        key_bits = []
        for i in range(n_bits):
            if alice_bases[i] == bob_bases[i]:
                key_bits.append(bob_bits[i])
        
        # Convert to string
        self._key = ''.join(str(b) for b in key_bits)
        self._raw_key = key_bits
        
        logger.info(f"BB84 completed. Raw key length: {len(self._raw_key)}")
        
        return self._key
    
    def get_key(self) -> str:
        """Get the sifted key"""
        return self._key
    
    def get_raw_key(self) -> List[int]:
        """Get the raw key (before sifting)"""
        return self._raw_key
    
    def get_basis_info(self) -> Dict[str, Any]:
        """
        Get information about bases used
        
        Returns:
            Dict: Basis information
        """
        return {
            'alice_bases': self._alice_bases.tolist() if hasattr(self, '_alice_bases') else [],
            'bob_bases': self._bob_bases.tolist() if hasattr(self, '_bob_bases') else [],
            'matching_bases': sum(1 for a, b in zip(
                self._alice_bases, self._bob_bases
            ) if a == b) if hasattr(self, '_alice_bases') else 0
        }


# ============================================================================
# SUPERDENSE CODING
# ============================================================================

class SuperdenseCoding:
    """
    Superdense coding protocol
    
    Sends 2 classical bits using 1 qubit by using entanglement.
    
    Example:
        >>> sc = SuperdenseCoding()
        >>> bits = sc.run((1, 0))
        >>> print(bits)  # (1, 0)
    """
    
    def __init__(self):
        """Initialize superdense coding protocol"""
        self.alice = Alice()
        self.bob = Bob()
        
        logger.info("Superdense coding initialized")
    
    def run(self, bits: Tuple[int, int]) -> Tuple[int, int]:
        """
        Run the superdense coding protocol
        
        Args:
            bits: Tuple of 2 bits to send (b1, b2)
            
        Returns:
            Tuple[int, int]: Decoded bits
            
        Example:
            >>> sc = SuperdenseCoding()
            >>> result = sc.run((1, 1))
            >>> print(result)  # (1, 1)
        """
        b1, b2 = bits
        
        if b1 not in [0, 1] or b2 not in [0, 1]:
            raise ValueError(f"Bits must be 0 or 1, got ({b1}, {b2})")
        
        logger.info(f"Superdense coding: sending ({b1}, {b2})")
        
        # Create Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        bell_state = bell_phi_plus()
        
        # Alice applies operations based on bits
        # 00: I ⊗ I (no change)
        # 01: X ⊗ I
        # 10: Z ⊗ I
        # 11: Z*X ⊗ I
        
        state = bell_state.data.copy()
        
        if b1 == 0 and b2 == 1:
            # Apply X on first qubit
            X = pauli_x()
            state = np.kron(X.data, identity(2).data) @ state
        elif b1 == 1 and b2 == 0:
            # Apply Z on first qubit
            Z = pauli_z()
            state = np.kron(Z.data, identity(2).data) @ state
        elif b1 == 1 and b2 == 1:
            # Apply X then Z on first qubit
            X = pauli_x()
            Z = pauli_z()
            state = np.kron(Z.data @ X.data, identity(2).data) @ state
        
        # Alice sends her qubit to Bob
        # Bob has both qubits now (his own + Alice's)
        
        # Bob applies CNOT then Hadamard to decode
        CNOT = cnot()
        H = hadamard()
        
        # Apply CNOT (control=first qubit, target=second qubit)
        state = CNOT.data @ state
        
        # Apply Hadamard on first qubit
        state = np.kron(H.data, identity(2).data) @ state
        
        # Measure
        probs = np.abs(state) ** 2
        probs = probs / np.sum(probs)
        
        outcome = int(np.random.choice(range(4), p=probs))
        
        # Decode bits from measurement
        # 0 -> 00, 1 -> 01, 2 -> 10, 3 -> 11
        decoded_b1 = outcome // 2
        decoded_b2 = outcome % 2
        
        logger.info(f"Superdense coding completed: received ({decoded_b1}, {decoded_b2})")
        
        return (decoded_b1, decoded_b2)


# ============================================================================
# QUANTUM TELEPORTATION
# ============================================================================

class QuantumTeleportation:
    """
    Quantum teleportation protocol
    
    Teleports a quantum state from Alice to Bob using entanglement.
    
    Example:
        >>> from psiqit.quantum import Ket
        >>> qt = QuantumTeleportation()
        >>> state = Ket([1/np.sqrt(2), 1/np.sqrt(2)])
        >>> result = qt.run(state)
        >>> print(result)  # Teleported state
    """
    
    def __init__(self):
        """Initialize quantum teleportation protocol"""
        self.alice = Alice()
        self.bob = Bob()
        
        logger.info("Quantum teleportation initialized")
    
    def run(self, state: Ket) -> Ket:
        """
        Run the quantum teleportation protocol
        
        Args:
            state: State to teleport (1 qubit)
            
        Returns:
            Ket: Teleported state
            
        Example:
            >>> from psiqit.quantum import Ket
            >>> qt = QuantumTeleportation()
            >>> original = Ket([1/np.sqrt(2), 1/np.sqrt(2)])
            >>> teleported = qt.run(original)
            >>> print(original == teleported)  # True (up to phase)
        """
        if state.dim != 2:
            raise ValueError(f"State must be 1 qubit, got dim={state.dim}")
        
        logger.info(f"Teleporting state: {state}")
        
        # Create Bell state |Φ⁺⟩
        bell = bell_phi_plus()
        
        # Alice has: state (qubit 0) + half of Bell (qubit 1)
        # Bob has: other half of Bell (qubit 2)
        
        # Combined state: |ψ⟩ ⊗ |Φ⁺⟩
        # This is a 3-qubit state
        dim = 2**3
        combined = np.zeros(dim, dtype=complex)
        
        # Construct the state manually
        # |ψ⟩ = a|0⟩ + b|1⟩
        a, b = state.data[0], state.data[1]
        
        # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        # Combined: (a|0⟩ + b|1⟩) ⊗ (|00⟩ + |11⟩)/√2
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # indices: qubit0=i, qubit1=j, qubit2=k
                    if j == k:  # Bell state condition
                        idx = i * 4 + j * 2 + k
                        amp = state.data[i] / np.sqrt(2)
                        combined[idx] = amp
        
        # Alice applies CNOT (control=qubit0, target=qubit1)
        # Then Hadamard on qubit0
        CNOT = cnot()
        H = hadamard()
        
        # Apply CNOT between qubit 0 and 1
        # CNOT on 3 qubits: I ⊗ CNOT (with control on qubit0)
        cnot_3 = np.kron(CNOT.data, identity(2).data)
        combined = cnot_3 @ combined
        
        # Apply Hadamard on qubit 0: H ⊗ I ⊗ I
        h_3 = np.kron(H.data, np.kron(identity(2).data, identity(2).data))
        combined = h_3 @ combined
        
        # Alice measures qubits 0 and 1
        # Project onto basis states
        probs = np.abs(combined) ** 2
        probs = probs / np.sum(probs)
        
        # Measurement outcomes for qubits 0 and 1
        outcomes = np.random.choice(range(4), p=probs)
        
        # Bob applies correction based on Alice's measurement
        # outcomes: 0->00, 1->01, 2->10, 3->11
        m1 = outcomes // 2
        m2 = outcomes % 2
        
        # Bob's state is qubit 2
        # Get the amplitude for qubit 2
        if m1 == 0 and m2 == 0:
            # Do nothing
            teleported = Ket(np.array([combined[outcomes], combined[outcomes + 4]]))
        elif m1 == 0 and m2 == 1:
            # Apply Z
            teleported = Ket(np.array([combined[outcomes], -combined[outcomes + 4]]))
        elif m1 == 1 and m2 == 0:
            # Apply X
            teleported = Ket(np.array([combined[outcomes + 4], combined[outcomes]]))
        else:  # m1 == 1 and m2 == 1
            # Apply Z then X
            teleported = Ket(np.array([-combined[outcomes + 4], combined[outcomes]]))
        
        # Normalize
        teleported = teleported.normalize()
        
        logger.info(f"Teleportation completed: {teleported}")
        
        return teleported


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'Alice',
    'Bob',
    'Charlie',
    'BB84',
    'SuperdenseCoding',
    'QuantumTeleportation',
] 
