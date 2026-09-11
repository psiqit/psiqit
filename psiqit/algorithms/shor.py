# psiqit/algorithms/shor.py

"""
Shor's Factoring Algorithm
Quantum algorithm for integer factorization
"""

import numpy as np
import math
import random
from typing import Optional, Dict, Any, List, Tuple, Union
from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, zero, one, plus, basis
from ..quantum.operator import Operator, identity, hadamard, phase, swap, cnot
from ..algorithms.qft import qft, iqft, qft_circuit
from ..algorithms.qpe import quantum_phase_estimation
from ..utils.logger import logger
from ..utils.validation import validate_qubits


# ============================================================================
# SHOR'S ALGORITHM - MAIN FUNCTION
# ============================================================================

def shor_factor(
    N: int,
    max_attempts: int = 10,
    use_quantum: bool = True,
    verbose: bool = False
) -> Tuple[int, int]:
    """
    Factor an integer N using Shor's algorithm
    
    Args:
        N: Integer to factor
        max_attempts: Maximum number of attempts
        use_quantum: Use quantum algorithm (if False, use classical simulation)
        verbose: Print progress information
        
    Returns:
        Tuple[int, int]: Factors (p, q) such that p * q = N
        
    Example:
        >>> factors = shor_factor(15)
        >>> print(factors)  # (3, 5)
        >>> factors = shor_factor(21)
        >>> print(factors)  # (3, 7)
        >>> factors = shor_factor(35)
        >>> print(factors)  # (5, 7)
    """
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    
    # Check if N is even
    if N % 2 == 0:
        logger.info(f"N={N} is even: {N} = 2 × {N//2}")
        return 2, N // 2
    
    # Check if N is a perfect power
    root = _is_perfect_power(N)
    if root is not None:
        logger.info(f"N={N} is a perfect power: {root} × {root}")
        return root, N // root
    
    for attempt in range(max_attempts):
        if verbose:
            logger.info(f"Attempt {attempt + 1}/{max_attempts}")
        
        # Choose random a such that 1 < a < N
        a = random.randint(2, N - 1)
        
        # Check gcd(a, N)
        g = math.gcd(a, N)
        if g > 1:
            logger.info(f"Found factor via gcd: {g}")
            return g, N // g
        
        if verbose:
            logger.info(f"a = {a}, gcd(a, N) = 1")
        
        # Find period r of f(x) = a^x mod N
        if use_quantum:
            r = _quantum_period_finding(a, N, verbose=verbose)
        else:
            r = _classical_period_finding(a, N)
        
        if verbose:
            logger.info(f"Period r = {r}")
        
        # Check if r is even and a^{r/2} ≠ -1 mod N
        if r % 2 == 0:
            x = pow(a, r // 2, N)
            if x != N - 1:
                # Found factors
                p = math.gcd(x - 1, N)
                q = math.gcd(x + 1, N)
                
                if p > 1 and q > 1 and p * q == N:
                    logger.info(f"Found factors: {p} × {q} = {N}")
                    return p, q
    
    raise RuntimeError(f"Failed to factor {N} after {max_attempts} attempts")


# ============================================================================
# PERIOD FINDING - QUANTUM
# ============================================================================

def _quantum_period_finding(
    a: int,
    N: int,
    verbose: bool = False
) -> int:
    """
    Find the period of f(x) = a^x mod N using quantum algorithm
    
    Args:
        a: Base
        N: Modulus
        verbose: Print progress
        
    Returns:
        int: Period r
    """
    # Determine number of qubits
    # We need enough qubits to represent numbers up to N^2
    n_qubits = int(np.ceil(np.log2(N * N)))
    n_qubits = max(n_qubits, 4)  # Minimum 4 qubits
    
    if verbose:
        logger.info(f"Period finding: n_qubits = {n_qubits}, a = {a}, N = {N}")
    
    # Create the circuit
    circ = QuantumCircuit(n_qubits)
    
    # Apply Hadamard to all qubits
    for i in range(n_qubits):
        circ.h(i)
    
    # Apply controlled-U operations
    # U|x⟩ = |a^x mod N⟩
    # We implement this using the modular exponentiation circuit
    # This is a simplified version - in practice, this is the most complex part
    
    # For simulation, we use classical period finding
    # This is because full modular exponentiation is complex to implement
    # in this simplified framework
    
    # Fallback to classical period finding for now
    logger.warning("Quantum period finding not fully implemented, using classical fallback")
    return _classical_period_finding(a, N)


def _modular_exponentiation_circuit(
    circuit: QuantumCircuit,
    a: int,
    N: int,
    control_qubits: List[int],
    target_qubits: List[int]
):
    """
    Build modular exponentiation circuit: |x⟩|0⟩ → |x⟩|a^x mod N⟩
    
    Args:
        circuit: QuantumCircuit object
        a: Base
        N: Modulus
        control_qubits: Qubits for x
        target_qubits: Qubits for output
    """
    # This is a placeholder for the full modular exponentiation circuit
    # In practice, this is the most complex part of Shor's algorithm
    # For now, we use a simplified version
    
    logger.warning("Modular exponentiation circuit is simplified")
    
    # For the simulation, we'll use classical computation
    # This is just a placeholder to make the circuit work


# ============================================================================
# PERIOD FINDING - CLASSICAL
# ============================================================================

def _classical_period_finding(a: int, N: int) -> int:
    """
    Find the period of f(x) = a^x mod N using classical algorithm
    
    Args:
        a: Base
        N: Modulus
        
    Returns:
        int: Period r
    """
    # Use the fact that the period is the multiplicative order of a modulo N
    # r = smallest r such that a^r ≡ 1 (mod N)
    
    r = 1
    current = a % N
    
    while current != 1:
        current = (current * a) % N
        r += 1
        
        # Safety check - avoid infinite loop
        if r > N:
            logger.warning(f"Period {r} exceeded N={N}")
            return r
    
    return r


# ============================================================================
# CLASSICAL SHOR'S ALGORITHM (SIMULATED)
# ============================================================================

def shor_classical(
    N: int,
    max_attempts: int = 10
) -> Tuple[int, int]:
    """
    Classical version of Shor's algorithm (for comparison)
    
    Args:
        N: Integer to factor
        max_attempts: Maximum number of attempts
        
    Returns:
        Tuple[int, int]: Factors
        
    Example:
        >>> factors = shor_classical(15)
        >>> print(factors)  # (3, 5)
    """
    logger.info(f"Classical Shor's algorithm for N={N}")
    
    for attempt in range(max_attempts):
        # Choose random a
        a = random.randint(2, N - 1)
        
        # Check gcd
        g = math.gcd(a, N)
        if g > 1:
            return g, N // g
        
        # Find period
        r = _classical_period_finding(a, N)
        
        # Check if r is even
        if r % 2 == 0:
            x = pow(a, r // 2, N)
            if x != N - 1:
                p = math.gcd(x - 1, N)
                q = math.gcd(x + 1, N)
                if p > 1 and q > 1 and p * q == N:
                    return p, q
    
    raise RuntimeError(f"Failed to factor {N}")


# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def _is_perfect_power(N: int) -> Optional[int]:
    """
    Check if N is a perfect power (N = a^b, b > 1)
    
    Args:
        N: Integer to check
        
    Returns:
        Optional[int]: Base a if perfect power, None otherwise
    """
    for b in range(2, int(np.log2(N)) + 1):
        a = int(round(N ** (1.0 / b)))
        if a ** b == N:
            return a
    return None


def shor_algorithm(
    N: int,
    use_quantum: bool = True,
    verbose: bool = True,
    max_attempts: int = 10
) -> Dict[str, Any]:
    """
    Run Shor's algorithm and return detailed results
    
    Args:
        N: Integer to factor
        use_quantum: Use quantum algorithm (if False, use classical simulation)
        verbose: Print progress
        max_attempts: Maximum number of attempts
        
    Returns:
        Dict: Detailed results
        
    Example:
        >>> result = shor_algorithm(15)
        >>> print(result['factors'])  # (3, 5)
        >>> print(result['attempts'])  # Number of attempts used
        >>> print(result['method'])  # 'quantum' or 'classical'
    """
    logger.info(f"Shor's algorithm for N={N}")
    
    attempts = 0
    factor_pairs = []
    
    # Check small factors first
    if N % 2 == 0:
        factor_pairs = [(2, N // 2)]
        attempts = 1
    
    # Check perfect powers
    root = _is_perfect_power(N)
    if root is not None and not factor_pairs:
        factor_pairs = [(root, N // root)]
        attempts = 1
    
    # Main algorithm
    if not factor_pairs:
        for attempt in range(max_attempts):
            attempts += 1
            a = random.randint(2, N - 1)
            g = math.gcd(a, N)
            
            if g > 1:
                factor_pairs.append((g, N // g))
                break
            
            if use_quantum:
                r = _quantum_period_finding(a, N, verbose=verbose)
            else:
                r = _classical_period_finding(a, N)
            
            if r % 2 == 0:
                x = pow(a, r // 2, N)
                if x != N - 1:
                    p = math.gcd(x - 1, N)
                    q = math.gcd(x + 1, N)
                    if p > 1 and q > 1 and p * q == N:
                        factor_pairs.append((p, q))
                        break
    
    if not factor_pairs:
        raise RuntimeError(f"Failed to factor {N} after {max_attempts} attempts")
    
    # Sort factors
    factors = tuple(sorted(factor_pairs[0]))
    
    return {
        'N': N,
        'factors': factors,
        'attempts': attempts,
        'method': 'quantum' if use_quantum else 'classical',
        'success': True,
        'algorithm': 'shor'
    }


# ========================================================================
# SHOR'S ALGORITHM STEPS (EXPLANATORY)
# ========================================================================

def shor_steps(N: int, a: int, r: int) -> Dict[str, Any]:
    """
    Show the steps of Shor's algorithm for a given a and period
    
    Args:
        N: Integer to factor
        a: Random base
        r: Period of a^x mod N
        
    Returns:
        Dict: Step-by-step results
        
    Example:
        >>> result = shor_steps(15, 7, 4)
        >>> print(result['step1'])  # "Choose a = 7, gcd(7, 15) = 1"
        >>> print(result['step2'])  # "Find period r = 4"
        >>> print(result['step3'])  # "r is even"
        >>> print(result['step4'])  # "x = 7^2 mod 15 = 4"
        >>> print(result['step5'])  # "Factors: gcd(3, 15) = 3, gcd(5, 15) = 5"
    """
    steps = {}
    
    steps['step1'] = f"Choose a = {a}, gcd({a}, {N}) = {math.gcd(a, N)}"
    
    if math.gcd(a, N) > 1:
        steps['step2'] = f"Found factor via gcd: {math.gcd(a, N)}"
        return steps
    
    steps['step2'] = f"Find period r of f(x) = {a}^x mod {N}: r = {r}"
    
    if r % 2 != 0:
        steps['step3'] = f"r = {r} is odd, try another a"
        return steps
    
    steps['step3'] = f"r = {r} is even"
    
    x = pow(a, r // 2, N)
    steps['step4'] = f"x = {a}^{r//2} mod {N} = {x}"
    
    if x == N - 1:
        steps['step5'] = f"x = {N-1} = -1 mod {N}, try another a"
        return steps
    
    p = math.gcd(x - 1, N)
    q = math.gcd(x + 1, N)
    
    steps['step5'] = f"Factors: gcd({x-1}, {N}) = {p}, gcd({x+1}, {N}) = {q}"
    
    if p * q == N:
        steps['step6'] = f"Success! {N} = {p} × {q}"
    else:
        steps['step6'] = f"Failed: {p} × {q} = {p*q} ≠ {N}"
    
    return steps


# ========================================================================
# EXPORTS
# ========================================================================

__all__ = [
    'shor_factor',
    'shor_classical',
    'shor_algorithm',
    'shor_steps',
]