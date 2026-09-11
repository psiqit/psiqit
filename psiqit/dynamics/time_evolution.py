# psiqit/dynamics/time_evolution.py


import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from ..quantum.state import Ket
from ..quantum.operator import Operator, identity, dagger
from ..math.qalgebra import commutator, expm
from ..utils.logger import logger
from ..utils.validation import is_hermitian


# ============================================================================
# EVOLUTION RESULT CLASS
# ============================================================================

@dataclass
class EvolutionResult:
    """
    Result container for time evolution
    
    Attributes:
        times: Time points
        states: List of quantum states at each time
        fidelities: List of fidelities with initial state
        success: Whether evolution was successful
        message: Additional information
        n_steps: Number of time steps
        final_state: Final state
        initial_state: Initial state
        method: Method used for evolution
    """
    times: np.ndarray
    states: List[Ket]
    fidelities: List[float]
    success: bool = True
    message: str = ""
    n_steps: int = 0
    final_state: Optional[Ket] = None
    initial_state: Optional[Ket] = None
    method: str = ""
    
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
    
    def get_fidelity_history(self) -> np.ndarray:
        """Get the fidelity history array"""
        return np.array(self.fidelities)
    
    def get_population(self, basis_state: int) -> List[float]:
        """
        Get population of a basis state over time
        
        Args:
            basis_state: Index of the basis state
            
        Returns:
            List[float]: Population values over time
        """
        populations = []
        for state in self.states:
            if basis_state < len(state.data):
                populations.append(float(np.abs(state.data[basis_state]) ** 2))
            else:
                populations.append(0.0)
        return populations
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"EvolutionResult(steps={self.n_steps}, status={status}, method={self.method})"
    
    def __str__(self) -> str:
        lines = [
            "Time Evolution Results:",
            f"  Method: {self.method}",
            f"  Steps: {self.n_steps}",
            f"  Success: {self.success}",
            f"  Message: {self.message}",
        ]
        if self.final_state is not None:
            lines.append(f"  Final Fidelity: {self.fidelities[-1]:.6f}" if self.fidelities else "  Final Fidelity: N/A")
        return "\n".join(lines)


# ============================================================================
# TROTTER EVOLUTION
# ============================================================================

class TrotterEvolution:
    """
    Trotter-Suzuki decomposition for quantum time evolution
    
    Implements the Trotter-Suzuki approximation:
    e^{-iHt} ≈ (e^{-iH₁t/n} e^{-iH₂t/n} ... e^{-iH_kt/n})^n
    
    Example:
        >>> from psiqit.quantum import pauli_x, pauli_z, zero
        >>> H = pauli_x() + pauli_z()
        >>> trotter = TrotterEvolution(H, n_steps=100, decomposition='first')
        >>> state = trotter.evolve(zero(), t=1.0)
        >>> print(state)
    """
    
    def __init__(
        self,
        hamiltonian: Operator,
        n_steps: int = 100,
        decomposition: str = 'first',
        hbar: float = 1.0
    ):
        """
        Initialize Trotter evolution
        
        Args:
            hamiltonian: Hamiltonian operator (can be sum of terms)
            n_steps: Number of Trotter steps
            decomposition: 'first' (first-order) or 'second' (second-order)
            hbar: Reduced Planck constant
        
        Example:
            >>> H = pauli_x() + pauli_z()
            >>> trotter = TrotterEvolution(H, n_steps=100, decomposition='first')
        """
        if not hamiltonian.is_hermitian:
            logger.warning("Hamiltonian is not Hermitian")
        
        self.hamiltonian = hamiltonian
        self.n_steps = n_steps
        self.decomposition = decomposition
        self.hbar = hbar
        self.dim = hamiltonian.dim
        
        # Decompose Hamiltonian into terms
        self.terms = self._decompose_hamiltonian(hamiltonian)
        
        logger.info(f"TrotterEvolution initialized: dim={self.dim}, steps={n_steps}, decomposition={decomposition}")
    
    def _decompose_hamiltonian(self, H: Operator) -> List[Operator]:
        """
        Decompose Hamiltonian into terms for Trotterization
        
        For now, we treat the Hamiltonian as a single term.
        In a more sophisticated implementation, we would split into
        kinetic and potential parts.
        
        Args:
            H: Hamiltonian operator
            
        Returns:
            List[Operator]: List of Hamiltonian terms
        """
        # Simple decomposition: treat as single term
        # For more complex Hamiltonians, this should be extended
        return [H]
    
    def _apply_trotter_step(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply a single Trotter step
        
        Args:
            psi: State vector
            dt: Time step
            
        Returns:
            np.ndarray: Evolved state
        """
        if self.decomposition == 'first':
            # First-order: e^{-iHdt} ≈ ∏ e^{-iH_i dt}
            for term in self.terms:
                U = expm(Operator(-1j * term.data * dt / self.hbar))
                psi = U.data @ psi
        elif self.decomposition == 'second':
            # Second-order: e^{-iHdt} ≈ e^{-iH_1 dt/2} e^{-iH_2 dt} e^{-iH_1 dt/2}
            # For two terms
            if len(self.terms) == 2:
                dt_half = dt / 2
                U1 = expm(Operator(-1j * self.terms[0].data * dt_half / self.hbar))
                U2 = expm(Operator(-1j * self.terms[1].data * dt / self.hbar))
                U3 = expm(Operator(-1j * self.terms[0].data * dt_half / self.hbar))
                psi = U1.data @ psi
                psi = U2.data @ psi
                psi = U3.data @ psi
            else:
                # Fallback to first-order for more terms
                logger.warning("Second-order Trotter only supports 2 terms, falling back to first-order")
                for term in self.terms:
                    U = expm(Operator(-1j * term.data * dt / self.hbar))
                    psi = U.data @ psi
        else:
            raise ValueError(f"Unknown decomposition: {self.decomposition}")
        
        return psi
    
    def evolve(self, psi0: Ket, t: float) -> Ket:
        """
        Evolve a state using Trotter decomposition
        
        Args:
            psi0: Initial state
            t: Evolution time
            
        Returns:
            Ket: Evolved state
        
        Example:
            >>> state = trotter.evolve(zero(), t=1.0)
        """
        if psi0.dim != self.dim:
            raise ValueError(f"State dimension {psi0.dim} does not match Hamiltonian dimension {self.dim}")
        
        logger.debug(f"Trotter evolution: t={t:.3f}, steps={self.n_steps}")
        
        dt = t / self.n_steps
        psi = psi0.data.copy()
        
        for step in range(self.n_steps):
            psi = self._apply_trotter_step(psi, dt)
            
            # Normalize
            norm = np.linalg.norm(psi)
            if norm > 0:
                psi = psi / norm
        
        logger.info(f"Trotter evolution completed: {self.n_steps} steps")
        
        return Ket(psi)
    
    def evolve_sequence(
        self,
        psi0: Ket,
        times: np.ndarray
    ) -> EvolutionResult:
        """
        Evolve a state at multiple time points
        
        Args:
            psi0: Initial state
            times: Array of time points
            
        Returns:
            EvolutionResult: Evolution results
        """
        logger.info(f"Trotter evolution sequence for {len(times)} time points")
        
        states = []
        fidelities = []
        success = True
        message = "Evolution completed successfully"
        
        try:
            for t in times:
                psi_t = self.evolve(psi0, t)
                states.append(psi_t)
                
                # Fidelity with initial state
                fid = abs(np.vdot(psi0.data, psi_t.data)) ** 2
                fidelities.append(float(fid))
        
        except Exception as e:
            success = False
            message = f"Evolution failed: {str(e)}"
            logger.error(message)
        
        return EvolutionResult(
            times=times,
            states=states,
            fidelities=fidelities,
            success=success,
            message=message,
            initial_state=psi0,
            method=f"Trotter ({self.decomposition})"
        )


# ============================================================================
# CHEBYSHEV EVOLUTION
# ============================================================================

class ChebyshevEvolution:
    """
    Chebyshev polynomial expansion for quantum time evolution
    
    Implements the Chebyshev polynomial method for time evolution:
    e^{-iHt} ψ ≈ Σ c_n T_n(H) ψ
    
    Example:
        >>> from psiqit.quantum import pauli_x, pauli_z, zero
        >>> H = pauli_x() + pauli_z()
        >>> cheb = ChebyshevEvolution(H, order=50)
        >>> state = cheb.evolve(zero(), t=1.0)
        >>> print(state)
    """
    
    def __init__(
        self,
        hamiltonian: Operator,
        order: int = 50,
        hbar: float = 1.0
    ):
        """
        Initialize Chebyshev evolution
        
        Args:
            hamiltonian: Hamiltonian operator
            order: Chebyshev polynomial order
            hbar: Reduced Planck constant
        
        Example:
            >>> H = pauli_x() + pauli_z()
            >>> cheb = ChebyshevEvolution(H, order=50)
        """
        if not hamiltonian.is_hermitian:
            logger.warning("Hamiltonian is not Hermitian")
        
        self.hamiltonian = hamiltonian
        self.order = order
        self.hbar = hbar
        self.dim = hamiltonian.dim
        
        # Rescale Hamiltonian to [-1, 1]
        self._rescale_hamiltonian()
        
        logger.info(f"ChebyshevEvolution initialized: dim={self.dim}, order={order}")
    
    def _rescale_hamiltonian(self):
        """
        Rescale Hamiltonian to [-1, 1] for Chebyshev expansion
        """
        # Compute eigenvalues to find spectral range
        eigvals = np.linalg.eigvalsh(self.hamiltonian.data)
        self.E_min = np.min(eigvals)
        self.E_max = np.max(eigvals)
        self.E_scale = (self.E_max - self.E_min) / 2
        self.E_shift = (self.E_max + self.E_min) / 2
        
        # Rescaled Hamiltonian: H_scaled = (H - E_shift) / E_scale
        self.H_scaled = (self.hamiltonian - self.E_shift * identity(self.dim)) / self.E_scale
        self.H_scaled.name = "H_scaled"
        
        logger.debug(f"Rescaled Hamiltonian: E_min={self.E_min:.3f}, E_max={self.E_max:.3f}")
    
    def _chebyshev_coefficients(self, t: float) -> List[complex]:
        """
        Compute Chebyshev coefficients for time evolution
        
        c_n = (2 - δ_{n0}) * (-i)^n * J_n(t)
        
        Args:
            t: Time
            
        Returns:
            List[complex]: Chebyshev coefficients
        """
        from scipy.special import jv
        
        # Scaled time
        tau = t * self.E_scale / self.hbar
        
        coefficients = []
        for n in range(self.order + 1):
            factor = 2 if n > 0 else 1
            coeff = factor * (-1j)**n * jv(n, tau)
            coefficients.append(coeff)
        
        return coefficients
    
    def evolve(self, psi0: Ket, t: float) -> Ket:
        """
        Evolve a state using Chebyshev polynomial expansion
        
        Args:
            psi0: Initial state
            t: Evolution time
            
        Returns:
            Ket: Evolved state
        
        Example:
            >>> state = cheb.evolve(zero(), t=1.0)
        """
        if psi0.dim != self.dim:
            raise ValueError(f"State dimension {psi0.dim} does not match Hamiltonian dimension {self.dim}")
        
        logger.debug(f"Chebyshev evolution: t={t:.3f}, order={self.order}")
        
        # Compute phase factor
        phase = np.exp(-1j * self.E_shift * t / self.hbar)
        
        # Compute Chebyshev coefficients
        coeffs = self._chebyshev_coefficients(t)
        
        # Initialize Chebyshev recursion
        psi = psi0.data.copy()
        
        # T_0(H_scaled) ψ = ψ
        psi_prev = psi0.data.copy()
        psi_curr = psi0.data.copy()
        
        # Accumulate result
        result = coeffs[0] * psi_prev
        
        # T_1(H_scaled) ψ = H_scaled ψ
        if self.order >= 1:
            psi_curr = self.H_scaled.data @ psi_prev
            result += coeffs[1] * psi_curr
        
        # Higher orders: T_{n+1} = 2 H_scaled T_n - T_{n-1}
        for n in range(2, self.order + 1):
            psi_next = 2 * self.H_scaled.data @ psi_curr - psi_prev
            result += coeffs[n] * psi_next
            psi_prev, psi_curr = psi_curr, psi_next
        
        # Apply phase factor
        result = phase * result
        
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        
        logger.info(f"Chebyshev evolution completed: order={self.order}")
        
        return Ket(result)
    
    def evolve_sequence(
        self,
        psi0: Ket,
        times: np.ndarray
    ) -> EvolutionResult:
        """
        Evolve a state at multiple time points
        
        Args:
            psi0: Initial state
            times: Array of time points
            
        Returns:
            EvolutionResult: Evolution results
        """
        logger.info(f"Chebyshev evolution sequence for {len(times)} time points")
        
        states = []
        fidelities = []
        success = True
        message = "Evolution completed successfully"
        
        try:
            for t in times:
                psi_t = self.evolve(psi0, t)
                states.append(psi_t)
                
                # Fidelity with initial state
                fid = abs(np.vdot(psi0.data, psi_t.data)) ** 2
                fidelities.append(float(fid))
        
        except Exception as e:
            success = False
            message = f"Evolution failed: {str(e)}"
            logger.error(message)
        
        return EvolutionResult(
            times=times,
            states=states,
            fidelities=fidelities,
            success=success,
            message=message,
            initial_state=psi0,
            method=f"Chebyshev (order={self.order})"
        )


# ============================================================================
# KRYLOV EVOLUTION
# ============================================================================

class KrylovEvolution:
    """
    Krylov subspace method for quantum time evolution
    
    Implements the Krylov subspace method for time evolution:
    Builds a Krylov subspace and projects the Hamiltonian,
    then evolves in the reduced subspace.
    
    Example:
        >>> from psiqit.quantum import pauli_x, pauli_z, zero
        >>> H = pauli_x() + pauli_z()
        >>> krylov = KrylovEvolution(H, krylov_dim=20)
        >>> state = krylov.evolve(zero(), t=1.0)
        >>> print(state)
    """
    
    def __init__(
        self,
        hamiltonian: Operator,
        krylov_dim: int = 20,
        hbar: float = 1.0
    ):
        """
        Initialize Krylov evolution
        
        Args:
            hamiltonian: Hamiltonian operator
            krylov_dim: Krylov subspace dimension
            hbar: Reduced Planck constant
        
        Example:
            >>> H = pauli_x() + pauli_z()
            >>> krylov = KrylovEvolution(H, krylov_dim=20)
        """
        if not hamiltonian.is_hermitian:
            logger.warning("Hamiltonian is not Hermitian")
        
        self.hamiltonian = hamiltonian
        self.krylov_dim = min(krylov_dim, hamiltonian.dim)
        self.hbar = hbar
        self.dim = hamiltonian.dim
        
        self._krylov_basis = None
        self._H_krylov = None
        
        logger.info(f"KrylovEvolution initialized: dim={self.dim}, krylov_dim={self.krylov_dim}")
    
    def _build_krylov_basis(self, psi0: np.ndarray) -> np.ndarray:
        """
        Build the Krylov subspace basis
        
        K = {ψ0, Hψ0, H²ψ0, ..., H^{m-1}ψ0}
        
        Args:
            psi0: Initial state vector
            
        Returns:
            np.ndarray: Krylov basis (columns are basis vectors)
        """
        m = min(self.krylov_dim, self.dim)
        
        # Initialize
        basis = np.zeros((self.dim, m), dtype=complex)
        basis[:, 0] = psi0 / np.linalg.norm(psi0)
        
        # Generate Krylov vectors
        for i in range(1, m):
            # Apply Hamiltonian to previous vector
            v = self.hamiltonian.data @ basis[:, i-1]
            
            # Orthogonalize against previous vectors
            for j in range(i):
                v = v - np.vdot(basis[:, j], v) * basis[:, j]
            
            # Normalize
            norm = np.linalg.norm(v)
            if norm > 1e-12:
                basis[:, i] = v / norm
            else:
                # Krylov subspace exhausted
                basis = basis[:, :i]
                break
        
        return basis
    
    def _project_hamiltonian(self, basis: np.ndarray) -> np.ndarray:
        """
        Project the Hamiltonian onto the Krylov subspace
        
        H_K = V† H V
        
        Args:
            basis: Krylov basis
            
        Returns:
            np.ndarray: Projected Hamiltonian
        """
        m = basis.shape[1]
        H_K = np.zeros((m, m), dtype=complex)
        
        for i in range(m):
            for j in range(m):
                H_K[i, j] = np.vdot(basis[:, i], self.hamiltonian.data @ basis[:, j])
        
        return H_K
    
    def evolve(self, psi0: Ket, t: float) -> Ket:
        """
        Evolve a state using Krylov subspace method
        
        Args:
            psi0: Initial state
            t: Evolution time
            
        Returns:
            Ket: Evolved state
        
        Example:
            >>> state = krylov.evolve(zero(), t=1.0)
        """
        if psi0.dim != self.dim:
            raise ValueError(f"State dimension {psi0.dim} does not match Hamiltonian dimension {self.dim}")
        
        logger.debug(f"Krylov evolution: t={t:.3f}, dim={self.krylov_dim}")
        
        # Build Krylov basis
        basis = self._build_krylov_basis(psi0.data)
        m = basis.shape[1]
        
        # Project Hamiltonian
        H_K = self._project_hamiltonian(basis)
        
        # Initial state in Krylov subspace
        psi0_K = np.zeros(m, dtype=complex)
        psi0_K[0] = 1.0
        
        # Evolve in Krylov subspace
        # Use matrix exponential
        U_K = expm(Operator(-1j * H_K * t / self.hbar))
        psi_K = U_K.data @ psi0_K
        
        # Transform back to full space
        psi = basis @ psi_K
        
        # Normalize
        norm = np.linalg.norm(psi)
        if norm > 0:
            psi = psi / norm
        
        logger.info(f"Krylov evolution completed: m={m}")
        
        return Ket(psi)
    
    def evolve_sequence(
        self,
        psi0: Ket,
        times: np.ndarray
    ) -> EvolutionResult:
        """
        Evolve a state at multiple time points
        
        Args:
            psi0: Initial state
            times: Array of time points
            
        Returns:
            EvolutionResult: Evolution results
        """
        logger.info(f"Krylov evolution sequence for {len(times)} time points")
        
        states = []
        fidelities = []
        success = True
        message = "Evolution completed successfully"
        
        try:
            for t in times:
                psi_t = self.evolve(psi0, t)
                states.append(psi_t)
                
                # Fidelity with initial state
                fid = abs(np.vdot(psi0.data, psi_t.data)) ** 2
                fidelities.append(float(fid))
        
        except Exception as e:
            success = False
            message = f"Evolution failed: {str(e)}"
            logger.error(message)
        
        return EvolutionResult(
            times=times,
            states=states,
            fidelities=fidelities,
            success=success,
            message=message,
            initial_state=psi0,
            method=f"Krylov (dim={self.krylov_dim})"
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'EvolutionResult',
    'TrotterEvolution',
    'ChebyshevEvolution',
    'KrylovEvolution',
]