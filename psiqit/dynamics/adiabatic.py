#psiqit/dynamics/adiabatic.py

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from ..quantum.state import Ket
from ..quantum.operator import Operator, identity, pauli_x, pauli_z
from ..math.qalgebra import PI
from ..math.ode_solver import rk4_system, ODEResult
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# ADIABATIC RESULT CLASS
# ============================================================================

@dataclass
class AdiabaticResult:
    """
    Result container for adiabatic evolution
    
    Attributes:
        times: Time points
        states: List of quantum states at each time
        energies: List of energies at each time
        fidelities: List of fidelities with instantaneous ground state
        success: Whether evolution was successful
        message: Additional information
        n_steps: Number of time steps
        final_state: Final state
        final_energy: Final energy
    """
    times: np.ndarray
    states: List[Ket]
    energies: List[float]
    fidelities: List[float]
    success: bool = True
    message: str = ""
    n_steps: int = 0
    final_state: Optional[Ket] = None
    final_energy: Optional[float] = None
    
    def __post_init__(self):
        """Post-process the result"""
        self.n_steps = len(self.times)
        
        if self.states:
            self.final_state = self.states[-1]
        
        if self.energies:
            self.final_energy = self.energies[-1]
    
    def get_fidelity_history(self) -> np.ndarray:
        """Get the fidelity history array"""
        return np.array(self.fidelities)
    
    def get_energy_history(self) -> np.ndarray:
        """Get the energy history array"""
        return np.array(self.energies)
    
    def get_state_at_time(self, index: int) -> Optional[Ket]:
        """Get state at a specific time index"""
        if 0 <= index < len(self.states):
            return self.states[index]
        return None
    
    def get_final_state(self) -> Optional[Ket]:
        """Get the final state"""
        return self.final_state
    
    def get_final_energy(self) -> Optional[float]:
        """Get the final energy"""
        return self.final_energy
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"AdiabaticResult(steps={self.n_steps}, status={status}, final_energy={self.final_energy:.6f if self.final_energy else 'N/A'})"
    
    def __str__(self) -> str:
        lines = [
            "Adiabatic Evolution Results:",
            f"  Steps: {self.n_steps}",
            f"  Success: {self.success}",
            f"  Message: {self.message}",
            f"  Final Energy: {self.final_energy:.8f}" if self.final_energy else "  Final Energy: N/A",
            f"  Final Fidelity: {self.fidelities[-1]:.6f}" if self.fidelities else "  Final Fidelity: N/A",
        ]
        return "\n".join(lines)


# ============================================================================
# ADIABATIC EVOLUTION
# ============================================================================

class AdiabaticEvolution:
    """
    Adiabatic quantum evolution
    
    Simulates the time-dependent Schrödinger equation for a Hamiltonian
    that interpolates between an initial and final Hamiltonian.
    
    H(t) = (1 - s(t)) * H_initial + s(t) * H_final
    
    where s(t) = t/T is the interpolation parameter.
    
    Example:
        >>> from psiqit.quantum import pauli_x, pauli_z, zero
        >>> H_i = pauli_x()  # Initial Hamiltonian
        >>> H_f = pauli_z()  # Final Hamiltonian
        >>> evo = AdiabaticEvolution(H_i, H_f, T=10.0, n_steps=100)
        >>> result = evo.evolve(zero())
        >>> print(result.final_energy)
    """
    
    def __init__(
        self,
        H_initial: Operator,
        H_final: Operator,
        T: float,
        n_steps: int = 100,
        schedule: Optional[Callable[[float], float]] = None
    ):
        """
        Initialize adiabatic evolution
        
        Args:
            H_initial: Initial Hamiltonian (at t=0)
            H_final: Final Hamiltonian (at t=T)
            T: Total evolution time
            n_steps: Number of time steps
            schedule: Interpolation schedule s(t) (default: linear s(t) = t/T)
        
        Example:
            >>> H_i = pauli_x()
            >>> H_f = pauli_z()
            >>> evo = AdiabaticEvolution(H_i, H_f, T=10.0, n_steps=100)
        """
        if H_initial.dim != H_final.dim:
            raise ValueError(f"Hamiltonian dimensions mismatch: {H_initial.dim} vs {H_final.dim}")
        
        self.H_initial = H_initial
        self.H_final = H_final
        self.T = T
        self.n_steps = n_steps
        self.dim = H_initial.dim
        
        # Default linear schedule
        if schedule is None:
            self.schedule = lambda t: t / T
        else:
            self.schedule = schedule
        
        logger.info(f"AdiabaticEvolution initialized: dim={self.dim}, T={T:.3f}, steps={n_steps}")
    
    def _hamiltonian(self, t: float) -> np.ndarray:
        """
        Get the Hamiltonian at time t
        
        Args:
            t: Time
            
        Returns:
            np.ndarray: Hamiltonian matrix
        """
        s = self.schedule(t)
        H = (1 - s) * self.H_initial.data + s * self.H_final.data
        return H
    
    def _get_ground_state(self, H: np.ndarray) -> Ket:
        """
        Get the ground state of a Hamiltonian
        
        Args:
            H: Hamiltonian matrix
            
        Returns:
            Ket: Ground state
        """
        eigvals, eigvecs = np.linalg.eigh(H)
        return Ket(eigvecs[:, 0])
    
    def _get_ground_energy(self, H: np.ndarray) -> float:
        """
        Get the ground state energy of a Hamiltonian
        
        Args:
            H: Hamiltonian matrix
            
        Returns:
            float: Ground state energy
        """
        eigvals = np.linalg.eigvalsh(H)
        return float(eigvals[0])
    
    def _schrodinger_derivative(self, t: float, psi_vec: np.ndarray) -> np.ndarray:
        """
        Compute dψ/dt = -i/ħ H(t) ψ
        
        Args:
            t: Time
            psi_vec: State vector (flattened)
            
        Returns:
            np.ndarray: Time derivative
        """
        H = self._hamiltonian(t)
        # Use hbar = 1 (natural units)
        return -1j * H @ psi_vec
    
    def evolve(self, psi0: Ket) -> AdiabaticResult:
        """
        Perform adiabatic evolution
        
        Args:
            psi0: Initial state (usually ground state of H_initial)
            
        Returns:
            AdiabaticResult: Evolution results
        
        Example:
            >>> from psiqit.quantum import zero
            >>> result = evo.evolve(zero())
            >>> print(result.final_energy)
        """
        if psi0.dim != self.dim:
            raise ValueError(f"State dimension {psi0.dim} does not match Hamiltonian dimension {self.dim}")
        
        logger.info(f"Starting adiabatic evolution for T={self.T:.3f}")
        
        # Time points
        times = np.linspace(0, self.T, self.n_steps + 1)
        dt = times[1] - times[0]
        
        # Initialize
        psi = psi0.data.copy()
        states = [psi0.copy()]
        energies = []
        fidelities = []
        
        # Track instantaneous ground state
        H0 = self._hamiltonian(0)
        ground_state = self._get_ground_state(H0)
        ground_energy = self._get_ground_energy(H0)
        
        # Initial fidelity
        initial_fidelity = abs(np.vdot(ground_state.data, psi)) ** 2
        fidelities.append(initial_fidelity)
        energies.append(ground_energy)
        
        # Evolution
        success = True
        message = "Evolution completed successfully"
        
        try:
            # Use RK4 for integration (or the system solver)
            # For complex state vectors, we need to handle real and imaginary parts separately
            # We'll use a simple RK4 implementation for complex vectors
            
            for i in range(self.n_steps):
                t = times[i]
                
                # RK4 for complex vector
                k1 = self._schrodinger_derivative(t, psi)
                k2 = self._schrodinger_derivative(t + dt/2, psi + dt/2 * k1)
                k3 = self._schrodinger_derivative(t + dt/2, psi + dt/2 * k2)
                k4 = self._schrodinger_derivative(t + dt, psi + dt * k3)
                
                psi_new = psi + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                
                # Normalize
                norm = np.linalg.norm(psi_new)
                if norm > 0:
                    psi_new = psi_new / norm
                else:
                    raise ValueError("State norm became zero")
                
                psi = psi_new
                
                # Store state
                states.append(Ket(psi.copy()))
                
                # Compute energy and fidelity at this time
                H_t = self._hamiltonian(times[i+1])
                energy = self._get_ground_energy(H_t)
                ground_state_t = self._get_ground_state(H_t)
                fidelity = abs(np.vdot(ground_state_t.data, psi)) ** 2
                
                energies.append(energy)
                fidelities.append(fidelity)
                
                # Check if fidelity is dropping too fast (potential diabatic transition)
                if i > 0 and fidelity < 0.1 and fidelities[i] > 0.9:
                    logger.warning(f"Sudden drop in fidelity at step {i}: {fidelities[i]:.3f} -> {fidelity:.3f}")
        
        except Exception as e:
            success = False
            message = f"Evolution failed: {str(e)}"
            logger.error(message)
        
        # Final results
        final_state = states[-1] if states else None
        final_energy = energies[-1] if energies else None
        
        logger.info(f"Adiabatic evolution completed: {message}")
        
        return AdiabaticResult(
            times=times,
            states=states,
            energies=energies,
            fidelities=fidelities,
            success=success,
            message=message,
            final_state=final_state,
            final_energy=final_energy
        )
    
    def get_min_gap(self, n_points: int = 1000) -> Tuple[float, float]:
        """
        Estimate the minimum energy gap during evolution
        
        Args:
            n_points: Number of points to sample
            
        Returns:
            Tuple: (min_gap, time_of_min_gap)
        
        Example:
            >>> evo = AdiabaticEvolution(H_i, H_f, T=10.0)
            >>> min_gap, t_gap = evo.get_min_gap()
            >>> print(f"Minimum gap: {min_gap:.6f} at t={t_gap:.3f}")
        """
        times = np.linspace(0, self.T, n_points)
        gaps = []
        
        for t in times:
            H = self._hamiltonian(t)
            eigvals = np.linalg.eigvalsh(H)
            gap = eigvals[1] - eigvals[0] if len(eigvals) > 1 else 0
            gaps.append(gap)
        
        min_gap = np.min(gaps)
        t_min_gap = times[np.argmin(gaps)]
        
        return float(min_gap), float(t_min_gap)
    
    def get_adiabatic_condition(self, n_points: int = 100) -> float:
        """
        Check the adiabatic condition: |⟨1|dH/dt|0⟩| / (ΔE)² << 1
        
        Args:
            n_points: Number of points to sample
            
        Returns:
            float: Maximum violation of adiabatic condition
        """
        times = np.linspace(0, self.T, n_points)
        max_ratio = 0.0
        
        for t in times:
            H = self._hamiltonian(t)
            eigvals, eigvecs = np.linalg.eigh(H)
            
            if len(eigvals) < 2:
                continue
            
            E0, E1 = eigvals[0], eigvals[1]
            delta_E = E1 - E0
            
            if delta_E < 1e-10:
                continue
            
            # Compute dH/dt
            dt = 1e-6
            H_plus = self._hamiltonian(t + dt)
            H_minus = self._hamiltonian(t - dt)
            dH = (H_plus - H_minus) / (2 * dt)
            
            # Matrix element ⟨1|dH/dt|0⟩
            psi0 = eigvecs[:, 0]
            psi1 = eigvecs[:, 1]
            
            matrix_element = abs(np.vdot(psi1, dH @ psi0))
            ratio = matrix_element / (delta_E ** 2)
            
            max_ratio = max(max_ratio, ratio)
        
        return float(max_ratio)


# ============================================================================
# QUANTUM ANNEALING
# ============================================================================

class QuantumAnnealing(AdiabaticEvolution):
    """
    Quantum annealing algorithm
    
    Quantum annealing is a special case of adiabatic evolution where
    the Hamiltonian is:
    
    H(s) = (1-s) * H_transverse + s * H_problem
    
    where H_transverse = -Σ σ_x^i (transverse field)
    and H_problem is the problem Hamiltonian (usually Ising model)
    
    Example:
        >>> # Simple 2-qubit Ising problem
        >>> problem_hamiltonian = {'Z0Z1': 1.0, 'Z0': 0.5, 'Z1': -0.5}
        >>> qa = QuantumAnnealing(problem_hamiltonian, T=10.0, n_steps=100)
        >>> result = qa.run()
        >>> print(result.final_energy)
    """
    
    def __init__(
        self,
        problem_hamiltonian: Dict[str, float],
        T: float,
        transverse_field: float = 1.0,
        n_steps: int = 100,
        schedule: Optional[Callable[[float], float]] = None
    ):
        """
        Initialize quantum annealing
        
        Args:
            problem_hamiltonian: Problem Hamiltonian as Pauli strings
                                 {'Z0': 1.0, 'Z0Z1': 0.5, ...}
            T: Total annealing time
            transverse_field: Strength of transverse field
            n_steps: Number of time steps
            schedule: Annealing schedule (default: linear)
        
        Example:
            >>> # Ising problem: -Z0Z1 (anti-ferromagnetic)
            >>> problem = {'Z0Z1': -1.0}
            >>> qa = QuantumAnnealing(problem, T=5.0, n_steps=50)
        """
        # Build the problem Hamiltonian as an Operator
        H_problem = self._build_hamiltonian(problem_hamiltonian)
        
        # Build the transverse field Hamiltonian
        # H_transverse = -Σ σ_x^i
        n_qubits = self._get_n_qubits_from_hamiltonian(problem_hamiltonian)
        H_transverse = self._build_transverse_hamiltonian(n_qubits, transverse_field)
        
        # Initialize the parent class
        super().__init__(H_transverse, H_problem, T, n_steps, schedule)
        
        self.problem_hamiltonian = problem_hamiltonian
        self.transverse_field = transverse_field
        self.n_qubits = n_qubits
        
        logger.info(f"QuantumAnnealing initialized: {n_qubits} qubits, T={T:.3f}")
    
    def _get_n_qubits_from_hamiltonian(self, hamiltonian: Dict[str, float]) -> int:
        """
        Extract the number of qubits from the Hamiltonian dictionary
        
        Args:
            hamiltonian: Hamiltonian dictionary
            
        Returns:
            int: Number of qubits
        """
        max_qubit = 0
        
        for term in hamiltonian.keys():
            # Parse the Pauli string (e.g., 'Z0Z1' -> qubits 0, 1)
            # Each character is a Pauli operator, followed by a number
            import re
            qubits = re.findall(r'\d+', term)
            for q in qubits:
                max_qubit = max(max_qubit, int(q))
        
        return max_qubit + 1
    
    def _build_hamiltonian(self, hamiltonian: Dict[str, float]) -> Operator:
        """
        Build an Operator from a Hamiltonian dictionary
        
        Args:
            hamiltonian: Dictionary of Pauli strings and coefficients
            
        Returns:
            Operator: Hamiltonian operator
        """
        from ..quantum.operator import pauli_string
        
        dim = 2 ** self._get_n_qubits_from_hamiltonian(hamiltonian)
        H = np.zeros((dim, dim), dtype=complex)
        
        for term, coeff in hamiltonian.items():
            if term == 'I':
                # Identity term
                H += coeff * np.eye(dim, dtype=complex)
            else:
                # Pauli string
                P = pauli_string(term)
                H += coeff * P.data
        
        return Operator(H, name="H_problem")
    
    def _build_transverse_hamiltonian(self, n_qubits: int, field: float) -> Operator:
        """
        Build the transverse field Hamiltonian: -field * Σ σ_x^i
        
        Args:
            n_qubits: Number of qubits
            field: Transverse field strength
            
        Returns:
            Operator: Transverse field Hamiltonian
        """
        from ..quantum.operator import pauli_x, tensor_product
        
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        
        for i in range(n_qubits):
            # Build σ_x^i
            X_i = np.eye(dim, dtype=complex)
            # Apply X on qubit i (simplified)
            # For each basis state, flip the i-th bit
            for j in range(dim):
                if (j >> i) & 1:
                    # Flip bit i
                    k = j ^ (1 << i)
                    X_i[k, j] = 1.0
                else:
                    k = j ^ (1 << i)
                    X_i[k, j] = 1.0
            
            H -= field * X_i
        
        return Operator(H, name="H_transverse")
    
    def run(
        self,
        initial_state: Optional[Ket] = None,
        return_all: bool = False
    ) -> AdiabaticResult:
        """
        Run the quantum annealing algorithm
        
        Args:
            initial_state: Initial state (default: ground state of transverse field)
            return_all: Return all states (for analysis)
            
        Returns:
            AdiabaticResult: Annealing results
        
        Example:
            >>> qa = QuantumAnnealing({'Z0Z1': -1.0}, T=5.0)
            >>> result = qa.run()
            >>> print(f"Final energy: {result.final_energy:.6f}")
            >>> print(f"Success: {result.success}")
        """
        # If no initial state is provided, use the ground state of H_transverse
        if initial_state is None:
            # Ground state of transverse field is |+⟩⊗n
            from ..quantum.state import plus
            state = plus()
            for _ in range(1, self.n_qubits):
                state = Ket(np.kron(state.data, plus().data))
            initial_state = state
        
        logger.info(f"Running quantum annealing for T={self.T:.3f}")
        
        # Perform adiabatic evolution
        result = super().evolve(initial_state)
        
        logger.info(f"Quantum annealing completed: final_energy={result.final_energy:.8f if result.final_energy else 'N/A'}")
        
        return result
    
    def get_final_state_probabilities(self, result: AdiabaticResult) -> Dict[str, float]:
        """
        Get the probabilities of the final state in the computational basis
        
        Args:
            result: Result from run()
            
        Returns:
            Dict: Basis state probabilities
        """
        if result.final_state is None:
            return {}
        
        probs = np.abs(result.final_state.data) ** 2
        
        # Create dictionary with binary labels
        labels = {}
        for i, p in enumerate(probs):
            binary = format(i, f'0{self.n_qubits}b')
            labels[binary] = float(p)
        
        return labels
    
    def get_ground_state_energy(self) -> float:
        """
        Get the ground state energy of the problem Hamiltonian
        
        Returns:
            float: Ground state energy
        """
        H = self._build_hamiltonian(self.problem_hamiltonian)
        eigvals = np.linalg.eigvalsh(H.data)
        return float(eigvals[0])
    
    def get_ground_state(self) -> Ket:
        """
        Get the ground state of the problem Hamiltonian
        
        Returns:
            Ket: Ground state
        """
        H = self._build_hamiltonian(self.problem_hamiltonian)
        eigvals, eigvecs = np.linalg.eigh(H.data)
        return Ket(eigvecs[:, 0])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ising_hamiltonian(
    J: Dict[Tuple[int, int], float],
    h: Dict[int, float] = None
) -> Dict[str, float]:
    """
    Create an Ising model Hamiltonian
    
    H = Σ J_ij Z_i Z_j + Σ h_i Z_i
    
    Args:
        J: Coupling strengths {(i, j): J_ij}
        h: Local fields {i: h_i}
        
    Returns:
        Dict: Hamiltonian as Pauli strings
        
    Example:
        >>> J = {(0, 1): -1.0}  # Ferromagnetic coupling
        >>> h = {0: 0.5, 1: -0.5}  # Local fields
        >>> H = ising_hamiltonian(J, h)
        >>> # H = -Z0Z1 + 0.5*Z0 - 0.5*Z1
    """
    hamiltonian = {}
    
    # Coupling terms
    for (i, j), Jij in J.items():
        term = f"Z{i}Z{j}"
        hamiltonian[term] = Jij
    
    # Local field terms
    if h:
        for i, hi in h.items():
            term = f"Z{i}"
            hamiltonian[term] = hamiltonian.get(term, 0) + hi
    
    return hamiltonian


def maxcut_hamiltonian(edges: List[Tuple[int, int]]) -> Dict[str, float]:
    """
    Create MaxCut problem Hamiltonian
    
    H = -Σ_{(i,j)∈E} Z_i Z_j
    
    Args:
        edges: List of edges in the graph
        
    Returns:
        Dict: Hamiltonian as Pauli strings
        
    Example:
        >>> edges = [(0, 1), (1, 2), (0, 2)]  # Triangle graph
        >>> H = maxcut_hamiltonian(edges)
        >>> # H = -Z0Z1 - Z1Z2 - Z0Z2
    """
    hamiltonian = {}
    
    for i, j in edges:
        term = f"Z{i}Z{j}"
        hamiltonian[term] = hamiltonian.get(term, 0) - 1.0
    
    return hamiltonian


def qubo_to_hamiltonian(
    qubo: Dict[Tuple[int, int], float],
    offset: float = 0.0
) -> Dict[str, float]:
    """
    Convert QUBO (Quadratic Unconstrained Binary Optimization) to Ising Hamiltonian
    
    QUBO: minimize Σ Q_ij x_i x_j, x_i ∈ {0, 1}
    Ising: H = Σ J_ij Z_i Z_j + Σ h_i Z_i + constant
    
    Mapping: x_i = (1 - Z_i)/2
    
    Args:
        qubo: QUBO matrix {(i, j): Q_ij}
        offset: Constant offset
        
    Returns:
        Dict: Hamiltonian as Pauli strings
        
    Example:
        >>> Q = {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0}
        >>> H = qubo_to_hamiltonian(Q)
    """
    hamiltonian = {}
    constant = offset
    
    for (i, j), Qij in qubo.items():
        if i == j:
            # Linear term: Q_ii * x_i = Q_ii * (1 - Z_i)/2
            # = Q_ii/2 - Q_ii/2 * Z_i
            constant += Qij / 2
            term = f"Z{i}"
            hamiltonian[term] = hamiltonian.get(term, 0) - Qij / 2
        else:
            # Quadratic term: Q_ij * x_i * x_j
            # = Q_ij/4 * (1 - Z_i - Z_j + Z_i Z_j)
            constant += Qij / 4
            term_i = f"Z{i}"
            hamiltonian[term_i] = hamiltonian.get(term_i, 0) - Qij / 4
            term_j = f"Z{j}"
            hamiltonian[term_j] = hamiltonian.get(term_j, 0) - Qij / 4
            term_ij = f"Z{i}Z{j}"
            hamiltonian[term_ij] = hamiltonian.get(term_ij, 0) + Qij / 4
    
    # Add constant term
    if abs(constant) > 1e-12:
        hamiltonian['I'] = constant
    
    return hamiltonian


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'AdiabaticResult',
    'AdiabaticEvolution',
    'QuantumAnnealing',
    'ising_hamiltonian',
    'maxcut_hamiltonian',
    'qubo_to_hamiltonian',
]