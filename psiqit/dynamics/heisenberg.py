#psiqit/dynamics/heisenberg.py


import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from ..quantum.operator import Operator, identity
from ..quantum.state import Ket
from ..math.qalgebra import commutator, dagger, trace, expm
from ..utils.logger import logger
from ..utils.validation import is_hermitian


# ============================================================================
# HEISENBERG RESULT CLASS
# ============================================================================

@dataclass
class HeisenbergResult:
    """
    Result container for Heisenberg evolution
    
    Attributes:
        times: Time points
        operators: List of operators at each time
        success: Whether evolution was successful
        message: Additional information
        n_steps: Number of time steps
        initial_operator: Initial operator
        final_operator: Final operator
        expectation_values: List of expectation values (if state provided)
    """
    times: np.ndarray
    operators: List[Operator]
    success: bool = True
    message: str = ""
    n_steps: int = 0
    initial_operator: Optional[Operator] = None
    final_operator: Optional[Operator] = None
    expectation_values: Optional[List[float]] = None
    
    def __post_init__(self):
        """Post-process the result"""
        self.n_steps = len(self.times)
        
        if self.operators:
            self.initial_operator = self.operators[0]
            self.final_operator = self.operators[-1]
    
    def get_operator_at_time(self, index: int) -> Optional[Operator]:
        """Get operator at a specific time index"""
        if 0 <= index < len(self.operators):
            return self.operators[index]
        return None
    
    def get_final_operator(self) -> Optional[Operator]:
        """Get the final operator"""
        return self.final_operator
    
    def get_expectation_history(self) -> Optional[np.ndarray]:
        """Get the expectation value history"""
        if self.expectation_values is not None:
            return np.array(self.expectation_values)
        return None
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"HeisenbergResult(steps={self.n_steps}, status={status})"
    
    def __str__(self) -> str:
        lines = [
            "Heisenberg Evolution Results:",
            f"  Steps: {self.n_steps}",
            f"  Success: {self.success}",
            f"  Message: {self.message}",
        ]
        if self.final_operator:
            lines.append(f"  Final Operator: {self.final_operator.name}")
        return "\n".join(lines)


# ============================================================================
# HEISENBERG EVOLUTION
# ============================================================================

class HeisenbergEvolution:
    """
    Heisenberg picture evolution of operators
    
    In the Heisenberg picture, operators evolve in time while states
    remain constant. The evolution of an operator A is given by:
    
    A(t) = e^{iHt/ħ} A(0) e^{-iHt/ħ}
    
    Example:
        >>> from psiqit.quantum import pauli_x, pauli_z
        >>> H = pauli_z()  # Hamiltonian
        >>> evo = HeisenbergEvolution(H)
        >>> X_t = evo.evolve_operator(pauli_x(), t=1.0)
        >>> print(X_t)  # Rotated operator
    """
    
    def __init__(self, hamiltonian: Operator, hbar: float = 1.0):
        """
        Initialize Heisenberg evolution
        
        Args:
            hamiltonian: Hamiltonian operator (Hermitian)
            hbar: Reduced Planck constant (default: 1)
        
        Example:
            >>> from psiqit.quantum import pauli_z
            >>> H = pauli_z()
            >>> evo = HeisenbergEvolution(H)
        """
        if not hamiltonian.is_hermitian:
            logger.warning("Hamiltonian is not Hermitian")
        
        self.hamiltonian = hamiltonian
        self.hbar = hbar
        self.dim = hamiltonian.dim
        
        logger.info(f"HeisenbergEvolution initialized: dim={self.dim}, hbar={hbar}")
    
    def evolve_operator(
        self,
        A: Operator,
        t: float,
        method: str = 'series',
        order: int = 5,
        n_steps: int = 100
    ) -> Operator:
        """
        Evolve an operator in the Heisenberg picture
        
        A(t) = e^{iHt/ħ} A(0) e^{-iHt/ħ}
        
        Args:
            A: Operator to evolve
            t: Evolution time
            method: 'series', 'exact', or 'trotter'
            order: Order for series expansion
            n_steps: Number of steps for Trotter method
            
        Returns:
            Operator: Evolved operator
        
        Example:
            >>> X = pauli_x()
            >>> X_t = evo.evolve_operator(X, t=np.pi/2)
            >>> print(X_t)  # Rotated operator
        """
        if A.dim != self.dim:
            raise ValueError(f"Operator dimension {A.dim} does not match Hamiltonian dimension {self.dim}")
        
        logger.debug(f"Evolving operator {A.name} for t={t:.3f} using {method}")
        
        if method == 'exact':
            # Exact evolution using matrix exponential
            H = self.hamiltonian.data
            U = expm(Operator(-1j * t * H / self.hbar))
            U_dag = U.dagger()
            
            # A(t) = U† A U
            A_evolved = U_dag @ A @ U
            A_evolved.name = f"{A.name}(t={t:.2f})"
            
            return A_evolved
        
        elif method == 'series':
            # Series expansion: A(t) = A + it/ħ [H, A] - t²/(2ħ²) [H, [H, A]] + ...
            A_t = Operator(A.data.copy(), name=A.name)
            
            # Compute terms iteratively
            term = A.data.copy()
            factor = 1.0
            
            for n in range(1, order + 1):
                # [H, term]
                term = commutator(Operator(H), Operator(term)).data
                factor *= (1j * t / self.hbar) / n
                
                A_t = A_t + Operator(factor * term)
            
            A_t.name = f"{A.name}(t={t:.2f})"
            
            return A_t
        
        elif method == 'trotter':
            # Trotter-Suzuki approximation
            # For small dt, evolve step by step
            dt = t / n_steps
            
            A_t = Operator(A.data.copy(), name=A.name)
            H = self.hamiltonian.data
            
            for _ in range(n_steps):
                # A(t+dt) = U†(dt) A(t) U(dt)
                # with U(dt) = exp(-iH dt/ħ)
                U = expm(Operator(-1j * H * dt / self.hbar))
                U_dag = U.dagger()
                A_t = U_dag @ A_t @ U
            
            A_t.name = f"{A.name}(t={t:.2f})"
            
            return A_t
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'exact', 'series', or 'trotter'")
    
    def evolve_operator_sequence(
        self,
        A: Operator,
        times: np.ndarray,
        method: str = 'exact',
        **kwargs
    ) -> HeisenbergResult:
        """
        Evolve an operator at multiple time points
        
        Args:
            A: Operator to evolve
            times: Array of time points
            method: Evolution method
            **kwargs: Additional arguments for evolve_operator
            
        Returns:
            HeisenbergResult: Evolution results
        
        Example:
            >>> times = np.linspace(0, 2*np.pi, 50)
            >>> result = evo.evolve_operator_sequence(pauli_x(), times)
            >>> print(len(result.operators))  # 50
        """
        logger.info(f"Evolving operator sequence for {len(times)} time points")
        
        operators = []
        success = True
        message = "Evolution completed successfully"
        
        try:
            for t in times:
                A_t = self.evolve_operator(A, t, method=method, **kwargs)
                operators.append(A_t)
        
        except Exception as e:
            success = False
            message = f"Evolution failed: {str(e)}"
            logger.error(message)
        
        return HeisenbergResult(
            times=times,
            operators=operators,
            success=success,
            message=message
        )
    
    def compute_expectation(
        self,
        A: Operator,
        state: Ket,
        t: float,
        method: str = 'exact',
        **kwargs
    ) -> float:
        """
        Compute expectation value ⟨ψ|A(t)|ψ⟩
        
        Args:
            A: Operator to evolve
            state: Quantum state
            t: Evolution time
            method: Evolution method
            **kwargs: Additional arguments for evolve_operator
            
        Returns:
            float: Expectation value
        
        Example:
            >>> from psiqit.quantum import zero
            >>> state = zero()
            >>> exp = evo.compute_expectation(pauli_x(), state, t=np.pi/2)
            >>> print(exp)
        """
        A_t = self.evolve_operator(A, t, method=method, **kwargs)
        expectation = np.vdot(state.data, A_t.data @ state.data).real
        return float(expectation)
    
    def compute_expectation_sequence(
        self,
        A: Operator,
        state: Ket,
        times: np.ndarray,
        method: str = 'exact',
        **kwargs
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Compute expectation values at multiple time points
        
        Args:
            A: Operator
            state: Quantum state
            times: Array of time points
            method: Evolution method
            **kwargs: Additional arguments for evolve_operator
            
        Returns:
            Tuple: (times, expectation_values)
        
        Example:
            >>> times = np.linspace(0, 2*np.pi, 50)
            >>> result = evo.compute_expectation_sequence(pauli_x(), zero(), times)
            >>> print(result[1][:5])  # First 5 expectation values
        """
        logger.info(f"Computing expectation sequence for {len(times)} time points")
        
        expectations = []
        
        for t in times:
            exp = self.compute_expectation(A, state, t, method=method, **kwargs)
            expectations.append(exp)
        
        return times, expectations


# ============================================================================
# HEISENBERG EQUATION
# ============================================================================

class HeisenbergEquation:
    """
    Heisenberg equation of motion
    
    dA/dt = (i/ħ) [H, A] + (∂A/∂t)
    
    For time-independent operators:
    dA/dt = (i/ħ) [H, A]
    
    Example:
        >>> from psiqit.quantum import pauli_x, pauli_z
        >>> H = pauli_z()
        >>> eq = HeisenbergEquation(H)
        >>> dXdt = eq.equation_of_motion(pauli_x())
        >>> print(dXdt)  # -i/ħ [Z, X] = -2/ħ Y
    """
    
    def __init__(self, hamiltonian: Operator, hbar: float = 1.0):
        """
        Initialize Heisenberg equation
        
        Args:
            hamiltonian: Hamiltonian operator
            hbar: Reduced Planck constant
        
        Example:
            >>> from psiqit.quantum import pauli_z
            >>> H = pauli_z()
            >>> eq = HeisenbergEquation(H)
        """
        if not hamiltonian.is_hermitian:
            logger.warning("Hamiltonian is not Hermitian")
        
        self.hamiltonian = hamiltonian
        self.hbar = hbar
        self.dim = hamiltonian.dim
        
        logger.info(f"HeisenbergEquation initialized: dim={self.dim}")
    
    def equation_of_motion(self, A: Operator) -> Operator:
        """
        Compute the Heisenberg equation of motion for an operator
        
        dA/dt = (i/ħ) [H, A]
        
        Args:
            A: Operator
            
        Returns:
            Operator: Time derivative of the operator
        
        Example:
            >>> X = pauli_x()
            >>> dXdt = eq.equation_of_motion(X)
            >>> print(dXdt)  # (i/ħ)[H, X]
        """
        if A.dim != self.dim:
            raise ValueError(f"Operator dimension {A.dim} does not match Hamiltonian dimension {self.dim}")
        
        # dA/dt = (i/ħ) [H, A]
        H = self.hamiltonian
        commutator_HA = H.commutator(A)
        dA_dt = (1j / self.hbar) * commutator_HA
        
        # Set name
        dA_dt.name = f"d{A.name}/dt" if A.name else "dA/dt"
        
        logger.debug(f"Equation of motion computed for {A.name}")
        return dA_dt
    
    def equation_of_motion_explicit(self, A: Operator) -> Operator:
        """
        Compute the explicit form of the equation of motion
        
        Same as equation_of_motion, but with expanded form
        
        Args:
            A: Operator
            
        Returns:
            Operator: Time derivative of the operator
        """
        return self.equation_of_motion(A)
    
    def get_time_evolution_operator(self, t: float) -> Operator:
        """
        Get the time evolution operator U(t) = e^{-iHt/ħ}
        
        Args:
            t: Time
            
        Returns:
            Operator: Time evolution operator
        """
        H = self.hamiltonian
        U = Operator(np.linalg.expm(-1j * H.data * t / self.hbar))
        U.name = f"U(t={t:.2f})"
        return U
    
    def evolve_operator_infinitesimal(
        self,
        A: Operator,
        dt: float
    ) -> Operator:
        """
        Evolve an operator by an infinitesimal time step
        
        A(t+dt) ≈ A(t) + (dA/dt) * dt
        
        Args:
            A: Operator
            dt: Time step
            
        Returns:
            Operator: Evolved operator
        """
        dA_dt = self.equation_of_motion(A)
        A_evolved = A + dt * dA_dt
        A_evolved.name = f"{A.name}(dt={dt:.3f})" if A.name else "A(t+dt)"
        return A_evolved
    
    def get_conserved_quantities(self, operators: List[Operator]) -> List[bool]:
        """
        Check which operators commute with the Hamiltonian
        
        Args:
            operators: List of operators
            
        Returns:
            List[bool]: True if operator commutes with Hamiltonian
        
        Example:
            >>> from psiqit.quantum import pauli_x, pauli_z, identity
            >>> H = pauli_z()
            >>> eq = HeisenbergEquation(H)
            >>> conserved = eq.get_conserved_quantities([pauli_x(), pauli_z(), identity()])
            >>> print(conserved)  # [False, True, True]
        """
        results = []
        
        for A in operators:
            commutator_HA = self.hamiltonian.commutator(A)
            is_conserved = np.allclose(commutator_HA.data, 0)
            results.append(is_conserved)
            
            if is_conserved:
                logger.debug(f"{A.name} is conserved")
            else:
                logger.debug(f"{A.name} is not conserved")
        
        return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def harmonic_oscillator_hamiltonian(
    omega: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0
) -> Operator:
    """
    Create harmonic oscillator Hamiltonian in position basis
    
    H = (p²/2m) + (1/2)mω²x²
    
    Note: This returns a simplified version using Pauli operators
          for a two-level approximation
    
    Args:
        omega: Angular frequency
        mass: Mass
        hbar: Reduced Planck constant
        
    Returns:
        Operator: Harmonic oscillator Hamiltonian
        
    Example:
        >>> H = harmonic_oscillator_hamiltonian(omega=1.0)
        >>> print(H.is_hermitian)  # True
    """
    from ..quantum.operator import pauli_x, pauli_y, pauli_z, identity
    
    # For a two-level system, use the Pauli representation
    # H = ħω/2 (Z) for a simple two-level system
    H = (hbar * omega / 2) * pauli_z()
    H.name = "H_harmonic"
    return H


def spin_hamiltonian(
    J: float,
    B: float = 0.0,
    n_spins: int = 2
) -> Operator:
    """
    Create spin Hamiltonian (Heisenberg model)
    
    H = -J Σ σ_i · σ_j - B Σ σ_z^i
    
    Args:
        J: Exchange coupling
        B: External magnetic field
        n_spins: Number of spins
        
    Returns:
        Operator: Spin Hamiltonian
        
    Example:
        >>> H = spin_hamiltonian(J=1.0, B=0.5, n_spins=2)
        >>> print(H.dim)  # 4
    """
    from ..quantum.operator import pauli_x, pauli_y, pauli_z, identity, tensor_product
    
    if n_spins < 2:
        raise ValueError(f"n_spins must be at least 2, got {n_spins}")
    
    dim = 2 ** n_spins
    H = np.zeros((dim, dim), dtype=complex)
    
    # Heisenberg term: -J Σ σ_i · σ_j
    for i in range(n_spins - 1):
        for j in range(i + 1, n_spins):
            # σ_i · σ_j = X_i X_j + Y_i Y_j + Z_i Z_j
            # Build X_i X_j
            X_iX_j = np.eye(dim, dtype=complex)
            # Simplified: for 2 spins, use direct construction
            if n_spins == 2:
                X = pauli_x().data
                Y = pauli_y().data
                Z = pauli_z().data
                
                # X⊗X + Y⊗Y + Z⊗Z
                H -= J * (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z))
            else:
                # For more spins, use tensor products
                # This is a simplified version
                logger.warning(f"Spin Hamiltonian for {n_spins} spins is simplified")
                X_iX_j = np.eye(dim, dtype=complex)
                # Placeholder for multi-spin case
                H -= J * X_iX_j
    
    # Zeeman term: -B Σ σ_z^i
    if abs(B) > 1e-12:
        for i in range(n_spins):
            Z_i = np.eye(dim, dtype=complex)
            # Simplified for 2 spins
            if n_spins == 2:
                if i == 0:
                    Z_i = np.kron(pauli_z().data, np.eye(2, dtype=complex))
                else:
                    Z_i = np.kron(np.eye(2, dtype=complex), pauli_z().data)
            H -= B * Z_i
    
    return Operator(H, name="H_spin")


def two_level_hamiltonian(
    delta: float,
    omega: float
) -> Operator:
    """
    Create a two-level Hamiltonian
    
    H = (δ/2) Z + (Ω/2) X
    
    Args:
        delta: Detuning (energy splitting)
        omega: Rabi frequency (coupling)
        
    Returns:
        Operator: Two-level Hamiltonian
        
    Example:
        >>> H = two_level_hamiltonian(delta=1.0, omega=0.5)
        >>> print(H.is_hermitian)  # True
    """
    from ..quantum.operator import pauli_x, pauli_z
    
    H = (delta / 2) * pauli_z() + (omega / 2) * pauli_x()
    H.name = "H_2level"
    return H


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'HeisenbergResult',
    'HeisenbergEvolution',
    'HeisenbergEquation',
    'harmonic_oscillator_hamiltonian',
    'spin_hamiltonian',
    'two_level_hamiltonian',
]