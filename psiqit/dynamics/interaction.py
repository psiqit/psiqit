# psiqit/dynamics/interaction.py


import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from ..quantum.operator import Operator, identity
from ..quantum.state import Ket
from ..math.qalgebra import commutator, expm, dagger
from ..utils.logger import logger
from ..utils.validation import is_hermitian


# ============================================================================
# INTERACTION RESULT CLASS
# ============================================================================

@dataclass
class InteractionResult:
    """
    Result container for interaction picture evolution
    
    Attributes:
        times: Time points
        states: List of quantum states at each time
        success: Whether evolution was successful
        message: Additional information
        n_steps: Number of time steps
        final_state: Final state
        initial_state: Initial state
    """
    times: np.ndarray
    states: List[Ket]
    success: bool = True
    message: str = ""
    n_steps: int = 0
    initial_state: Optional[Ket] = None
    final_state: Optional[Ket] = None
    
    def __post_init__(self):
        """Post-process the result"""
        self.n_steps = len(self.times)
        
        if self.states:
            self.initial_state = self.states[0]
            self.final_state = self.states[-1]
    
    def get_state_at_time(self, index: int) -> Optional[Ket]:
        """Get state at a specific time index"""
        if 0 <= index < len(self.states):
            return self.states[index]
        return None
    
    def get_final_state(self) -> Optional[Ket]:
        """Get the final state"""
        return self.final_state
    
    def get_fidelity_with_initial(self) -> List[float]:
        """
        Get fidelity of each state with the initial state
        
        Returns:
            List[float]: Fidelity values
        """
        if not self.states or self.initial_state is None:
            return []
        
        fidelities = []
        for state in self.states:
            fid = abs(np.vdot(self.initial_state.data, state.data)) ** 2
            fidelities.append(float(fid))
        
        return fidelities
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"InteractionResult(steps={self.n_steps}, status={status})"
    
    def __str__(self) -> str:
        lines = [
            "Interaction Picture Evolution Results:",
            f"  Steps: {self.n_steps}",
            f"  Success: {self.success}",
            f"  Message: {self.message}",
        ]
        if self.final_state:
            lines.append(f"  Final State: {self.final_state}")
        return "\n".join(lines)


# ============================================================================
# INTERACTION PICTURE
# ============================================================================

class InteractionPicture:
    """
    Interaction picture dynamics
    
    In the interaction picture, the Hamiltonian is split into:
    H = H₀ + V
    
    where H₀ is the free (solvable) part and V is the interaction.
    The evolution is governed by the interaction Hamiltonian:
    
    V_I(t) = e^{iH₀t/ħ} V e^{-iH₀t/ħ}
    
    States evolve as:
    |ψ_I(t)⟩ = U_I(t) |ψ_I(0)⟩
    
    where U_I(t) = T exp(-i/ħ ∫₀ᵗ V_I(t') dt')
    
    Example:
        >>> from psiqit.quantum import pauli_z, pauli_x
        >>> H0 = pauli_z()  # Free Hamiltonian
        >>> V = pauli_x()   # Interaction
        >>> ip = InteractionPicture(H0, V)
        >>> psi_t = ip.evolve_state(zero(), t=1.0)
    """
    
    def __init__(self, H0: Operator, V: Operator, hbar: float = 1.0):
        """
        Initialize interaction picture
        
        Args:
            H0: Free Hamiltonian (solvable part)
            V: Interaction Hamiltonian
            hbar: Reduced Planck constant
        
        Example:
            >>> from psiqit.quantum import pauli_z, pauli_x
            >>> H0 = pauli_z()
            >>> V = pauli_x()
            >>> ip = InteractionPicture(H0, V)
        """
        if H0.dim != V.dim:
            raise ValueError(f"Hamiltonian dimensions mismatch: {H0.dim} vs {V.dim}")
        
        if not H0.is_hermitian:
            logger.warning("H0 is not Hermitian")
        
        if not V.is_hermitian:
            logger.warning("V is not Hermitian")
        
        self.H0 = H0
        self.V = V
        self.hbar = hbar
        self.dim = H0.dim
        
        # Cache time evolution operators for efficiency
        self._U0_cache = {}
        
        logger.info(f"InteractionPicture initialized: dim={self.dim}, hbar={hbar}")
    
    def _get_U0(self, t: float) -> Operator:
        """
        Get the free evolution operator U₀(t) = e^{-iH₀t/ħ}
        
        Args:
            t: Time
            
        Returns:
            Operator: Free evolution operator
        """
        # Check cache
        if t in self._U0_cache:
            return self._U0_cache[t]
        
        # Compute U0
        H0 = self.H0
        U0 = expm(Operator(-1j * H0.data * t / self.hbar))
        U0.name = f"U0(t={t:.2f})"
        
        # Cache
        self._U0_cache[t] = U0
        
        return U0
    
    def _get_U0_dagger(self, t: float) -> Operator:
        """
        Get the adjoint of the free evolution operator
        
        Args:
            t: Time
            
        Returns:
            Operator: U₀†(t)
        """
        return self._get_U0(t).dagger()
    
    def interaction_operator(self, t: float) -> Operator:
        """
        Compute the interaction Hamiltonian in the interaction picture
        
        V_I(t) = e^{iH₀t/ħ} V e^{-iH₀t/ħ}
        
        Args:
            t: Time
            
        Returns:
            Operator: Interaction operator V_I(t)
        
        Example:
            >>> ip = InteractionPicture(H0, V)
            >>> V_I = ip.interaction_operator(t=0.5)
            >>> print(V_I.name)
        """
        U0_dag = self._get_U0_dagger(t)
        U0 = self._get_U0(t)
        
        V_I = U0_dag @ self.V @ U0
        V_I.name = f"V_I(t={t:.2f})"
        
        return V_I
    
    def _schrodinger_derivative(self, t: float, psi_vec: np.ndarray) -> np.ndarray:
        """
        Compute dψ/dt = -i/ħ V_I(t) ψ
        
        Args:
            t: Time
            psi_vec: State vector (flattened)
            
        Returns:
            np.ndarray: Time derivative
        """
        V_I = self.interaction_operator(t)
        return -1j * V_I.data @ psi_vec / self.hbar
    
    def evolve_state(
        self,
        psi0: Ket,
        t: float,
        dt: float = 0.01,
        method: str = 'rk4'
    ) -> Ket:
        """
        Evolve a state in the interaction picture
        
        |ψ_I(t)⟩ = U_I(t) |ψ_I(0)⟩
        
        Args:
            psi0: Initial state (in interaction picture)
            t: Evolution time
            dt: Time step
            method: Integration method ('euler' or 'rk4')
            
        Returns:
            Ket: Evolved state
        
        Example:
            >>> from psiqit.quantum import zero
            >>> psi_t = ip.evolve_state(zero(), t=1.0, dt=0.01)
            >>> print(psi_t)
        """
        if psi0.dim != self.dim:
            raise ValueError(f"State dimension {psi0.dim} does not match Hamiltonian dimension {self.dim}")
        
        logger.debug(f"Evolving state for t={t:.3f} with dt={dt:.4f}")
        
        # Number of steps
        n_steps = int(np.ceil(t / dt))
        dt_actual = t / n_steps
        
        # Initialize
        psi = psi0.data.copy()
        
        # Evolution
        for step in range(n_steps):
            time = step * dt_actual
            
            if method == 'euler':
                # Euler method
                dpsi = self._schrodinger_derivative(time, psi)
                psi = psi + dt_actual * dpsi
            
            elif method == 'rk4':
                # RK4 method
                k1 = self._schrodinger_derivative(time, psi)
                k2 = self._schrodinger_derivative(time + dt_actual/2, psi + dt_actual/2 * k1)
                k3 = self._schrodinger_derivative(time + dt_actual/2, psi + dt_actual/2 * k2)
                k4 = self._schrodinger_derivative(time + dt_actual, psi + dt_actual * k3)
                
                psi = psi + dt_actual/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            else:
                raise ValueError(f"Unknown method: {method}. Use 'euler' or 'rk4'")
            
            # Normalize
            norm = np.linalg.norm(psi)
            if norm > 0:
                psi = psi / norm
            else:
                logger.warning(f"State norm became zero at step {step}")
                break
        
        return Ket(psi)
    
    def evolve_state_sequence(
        self,
        psi0: Ket,
        times: np.ndarray,
        dt: float = 0.01,
        method: str = 'rk4'
    ) -> InteractionResult:
        """
        Evolve a state at multiple time points
        
        Args:
            psi0: Initial state
            times: Array of time points
            dt: Time step for integration
            method: Integration method
            
        Returns:
            InteractionResult: Evolution results
        
        Example:
            >>> times = np.linspace(0, 2*np.pi, 50)
            >>> result = ip.evolve_state_sequence(zero(), times)
            >>> print(len(result.states))  # 50
        """
        logger.info(f"Evolving state sequence for {len(times)} time points")
        
        states = []
        success = True
        message = "Evolution completed successfully"
        
        try:
            for t in times:
                psi_t = self.evolve_state(psi0, t, dt=dt, method=method)
                states.append(psi_t)
        
        except Exception as e:
            success = False
            message = f"Evolution failed: {str(e)}"
            logger.error(message)
        
        return InteractionResult(
            times=times,
            states=states,
            success=success,
            message=message,
            initial_state=psi0
        )
    
    def evolve_operator(self, A: Operator, t: float) -> Operator:
        """
        Evolve an operator in the interaction picture
        
        A_I(t) = e^{iH₀t/ħ} A e^{-iH₀t/ħ}
        
        Args:
            A: Operator to evolve
            t: Time
            
        Returns:
            Operator: Evolved operator
        
        Example:
            >>> from psiqit.quantum import pauli_x
            >>> A_t = ip.evolve_operator(pauli_x(), t=1.0)
            >>> print(A_t.name)
        """
        if A.dim != self.dim:
            raise ValueError(f"Operator dimension {A.dim} does not match Hamiltonian dimension {self.dim}")
        
        U0_dag = self._get_U0_dagger(t)
        U0 = self._get_U0(t)
        
        A_I = U0_dag @ A @ U0
        A_I.name = f"{A.name}_I(t={t:.2f})" if A.name else f"A_I(t={t:.2f})"
        
        return A_I
    
    def evolve_operator_sequence(
        self,
        A: Operator,
        times: np.ndarray
    ) -> List[Operator]:
        """
        Evolve an operator at multiple time points
        
        Args:
            A: Operator
            times: Array of time points
            
        Returns:
            List[Operator]: Evolved operators
        """
        logger.info(f"Evolving operator sequence for {len(times)} time points")
        
        operators = []
        
        for t in times:
            A_t = self.evolve_operator(A, t)
            operators.append(A_t)
        
        return operators
    
    def compute_expectation(
        self,
        A: Operator,
        state: Ket,
        t: float
    ) -> float:
        """
        Compute expectation value ⟨ψ|A_I(t)|ψ⟩
        
        Args:
            A: Operator
            state: Quantum state
            t: Time
            
        Returns:
            float: Expectation value
        """
        A_I = self.evolve_operator(A, t)
        expectation = np.vdot(state.data, A_I.data @ state.data).real
        return float(expectation)
    
    def compute_expectation_sequence(
        self,
        A: Operator,
        state: Ket,
        times: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Compute expectation values at multiple time points
        
        Args:
            A: Operator
            state: Quantum state
            times: Array of time points
            
        Returns:
            Tuple: (times, expectation_values)
        """
        logger.info(f"Computing expectation sequence for {len(times)} time points")
        
        expectations = []
        
        for t in times:
            exp = self.compute_expectation(A, state, t)
            expectations.append(exp)
        
        return times, expectations
    
    def get_time_evolution_operator(self, t: float) -> Operator:
        """
        Get the full time evolution operator in the interaction picture
        
        U_I(t) = T exp(-i/ħ ∫₀ᵗ V_I(t') dt')
        
        Args:
            t: Time
            
        Returns:
            Operator: Time evolution operator
        
        Note:
            This is a simplified version that uses the series expansion
            for the time-ordered exponential.
        """
        # For small t, use series expansion
        # U_I(t) ≈ I - i/ħ ∫₀ᵗ V_I(t') dt'
        # This is a first-order approximation
        
        # Integrate V_I numerically
        n_points = 100
        dt = t / n_points
        integral = np.zeros((self.dim, self.dim), dtype=complex)
        
        for i in range(n_points):
            t_i = i * dt
            V_I = self.interaction_operator(t_i)
            integral += V_I.data * dt
        
        U_I = np.eye(self.dim, dtype=complex) - 1j * integral / self.hbar
        
        return Operator(U_I, name=f"U_I(t={t:.2f})")
    
    def to_schrodinger_picture(
        self,
        psi_I: Ket,
        t: float
    ) -> Ket:
        """
        Convert a state from interaction picture to Schrödinger picture
        
        |ψ_S(t)⟩ = e^{-iH₀t/ħ} |ψ_I(t)⟩
        
        Args:
            psi_I: State in interaction picture
            t: Time
            
        Returns:
            Ket: State in Schrödinger picture
        """
        U0 = self._get_U0(t)
        psi_S = U0 @ psi_I
        return Ket(psi_S.data)
    
    def to_interaction_picture(
        self,
        psi_S: Ket,
        t: float
    ) -> Ket:
        """
        Convert a state from Schrödinger picture to interaction picture
        
        |ψ_I(t)⟩ = e^{iH₀t/ħ} |ψ_S(t)⟩
        
        Args:
            psi_S: State in Schrödinger picture
            t: Time
            
        Returns:
            Ket: State in interaction picture
        """
        U0_dag = self._get_U0_dagger(t)
        psi_I = U0_dag @ psi_S
        return Ket(psi_I.data)
    
    def clear_cache(self):
        """Clear the cache of time evolution operators"""
        self._U0_cache = {}
        logger.debug("Cache cleared")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_interaction_hamiltonian(
    coupling: float,
    operator_A: Operator,
    operator_B: Operator
) -> Operator:
    """
    Create an interaction Hamiltonian of the form V = g * A ⊗ B
    
    Args:
        coupling: Coupling strength g
        operator_A: Operator acting on system A
        operator_B: Operator acting on system B
        
    Returns:
        Operator: Interaction Hamiltonian
        
    Example:
        >>> from psiqit.quantum import pauli_x, pauli_z
        >>> V = create_interaction_hamiltonian(0.5, pauli_x(), pauli_z())
        >>> print(V.name)
    """
    from ..quantum.operator import tensor_product
    
    V = coupling * tensor_product(operator_A, operator_B)
    V.name = f"V_interaction(g={coupling:.2f})"
    return V


def rotating_frame(
    H0: Operator,
    V: Operator,
    omega: float,
    t: float
) -> Operator:
    """
    Transform to a rotating frame
    
    In the rotating frame, the Hamiltonian becomes:
    H_rot = e^{iωt/ħ} H e^{-iωt/ħ} - ħω I
    
    Args:
        H0: Free Hamiltonian
        V: Interaction Hamiltonian
        omega: Rotation frequency
        t: Time
        
    Returns:
        Operator: Hamiltonian in the rotating frame
    """
    from ..quantum.operator import identity
    
    # Rotation operator R = e^{-iωt/ħ} (for spin systems)
    # This is a simplified version
    
    # For a two-level system, rotation is achieved by R = e^{-iωt/2} Z
    # The rotating frame Hamiltonian:
    # H_rot = (δ/2) Z + (Ω/2) X
    
    # This is a placeholder for a full implementation
    H_rot = H0 + V
    H_rot.name = "H_rotating"
    
    return H_rot


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'InteractionResult',
    'InteractionPicture',
    'create_interaction_hamiltonian',
    'rotating_frame',
]
