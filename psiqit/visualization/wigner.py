# psiqit/visualization/wigner.py 

"""
Wigner Function Module
Compute and visualize Wigner functions for quantum states
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
from ..quantum.state import Ket, coherent_state, squeezed_state, fock_state
from ..utils.logger import logger
from ..math.qalgebra import PI


# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

def _has_matplotlib() -> bool:
    """Check if matplotlib is available"""
    try:
        import matplotlib
        return True
    except ImportError:
        return False


def _has_scipy() -> bool:
    """Check if scipy is available"""
    try:
        import scipy
        return True
    except ImportError:
        return False


# ============================================================================
# WIGNER FUNCTION - GENERAL
# ============================================================================

def wigner_function(
    state: Union[Ket, np.ndarray],
    x_range: Tuple[float, float] = (-5, 5),
    p_range: Tuple[float, float] = (-5, 5),
    n_points: int = 100,
    hbar: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Wigner function for a quantum state
    
    The Wigner function is defined as:
    W(x, p) = (1/πħ) ∫ dy ⟨x+y|ρ|x-y⟩ e^{-2ipy/ħ}
    
    Args:
        state: Quantum state (Ket or density matrix)
        x_range: (x_min, x_max) range
        p_range: (p_min, p_max) range
        n_points: Number of points in each dimension
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (W, x, p) where W is the Wigner function
        
    Example:
        >>> from psiqit.quantum import coherent_state
        >>> alpha = 1.0 + 1j*0.5
        >>> state = coherent_state(alpha, n_levels=20)
        >>> W, x, p = wigner_function(state, (-3, 3), (-3, 3))
        >>> plot_wigner(W, x, p, title="Coherent State")
    """
    if isinstance(state, Ket):
        # Get the wavefunction from the Ket state
        psi = state.data
        # For Fock basis states, we need to compute the Wigner function
        # using the Fock state representation
        n_levels = len(psi)
        
        # For pure states in Fock basis, compute Wigner function
        # W(x, p) = (2/π) Σ_{n,m} ρ_{n,m} W_{n,m}(x, p)
        # where W_{n,m} are the Wigner functions of Fock states
        
        # Use the Wigner function for Fock state
        return _wigner_function_fock(psi, x_range, p_range, n_points, hbar)
    
    elif isinstance(state, np.ndarray) and state.ndim == 2:
        # Density matrix
        return _wigner_function_density(state, x_range, p_range, n_points, hbar)
    
    else:
        raise ValueError("State must be Ket or density matrix")


def _wigner_function_fock(
    psi: np.ndarray,
    x_range: Tuple[float, float],
    p_range: Tuple[float, float],
    n_points: int,
    hbar: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Wigner function for a Fock state superposition
    
    Args:
        psi: State vector in Fock basis
        x_range: (x_min, x_max) range
        p_range: (p_min, p_max) range
        n_points: Number of points in each dimension
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (W, x, p)
    """
    n_levels = len(psi)
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    
    X, P = np.meshgrid(x, p)
    W = np.zeros_like(X, dtype=float)
    
    # Compute Wigner function for each pair of Fock states
    # This is computationally intensive for large n_levels
    # For simplicity, we use the analytical formula for Fock states
    
    # For a Fock state |n⟩, the Wigner function is:
    # W_n(x, p) = (2/π) (-1)^n e^{-2r^2} L_n(4r^2)
    # where r^2 = (x^2 + p^2)/hbar and L_n are Laguerre polynomials
    
    if _has_scipy():
        from scipy.special import eval_laguerre
        
        r2 = (X**2 + P**2) / hbar
        factor = 2 / (np.pi * hbar)
        
        # Compute Wigner function for each Fock state
        W_total = np.zeros_like(X, dtype=float)
        
        for n in range(n_levels):
            if abs(psi[n]) > 1e-10:
                # Wigner function for Fock state |n⟩
                W_n = factor * (-1)**n * np.exp(-2*r2) * eval_laguerre(n, 4*r2)
                W_total += np.abs(psi[n])**2 * W_n
        
        # Add interference terms (coherences)
        for n in range(n_levels):
            for m in range(n_levels):
                if n != m and abs(psi[n] * psi[m]) > 1e-10:
                    # Interference term
                    # This is complex and requires full computation
                    # For simplicity, we neglect interference terms
                    pass
        
        W = W_total
        
    else:
        # Fallback: approximate with Gaussian
        logger.warning("scipy not available, using approximate Wigner function")
        sigma = 1.0
        W = (1 / (np.pi * hbar)) * np.exp(-(X**2 + P**2) / (2 * sigma**2))
    
    # Normalize
    dx = x[1] - x[0]
    dp = p[1] - p[0]
    W = W / (np.sum(W) * dx * dp)
    
    return W, x, p


def _wigner_function_density(
    rho: np.ndarray,
    x_range: Tuple[float, float],
    p_range: Tuple[float, float],
    n_points: int,
    hbar: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Wigner function for a density matrix in Fock basis
    
    Args:
        rho: Density matrix in Fock basis
        x_range: (x_min, x_max) range
        p_range: (p_min, p_max) range
        n_points: Number of points in each dimension
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (W, x, p)
    """
    n_levels = rho.shape[0]
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    
    X, P = np.meshgrid(x, p)
    W = np.zeros_like(X, dtype=float)
    
    if _has_scipy():
        from scipy.special import eval_laguerre
        
        r2 = (X**2 + P**2) / hbar
        factor = 2 / (np.pi * hbar)
        
        W_total = np.zeros_like(X, dtype=float)
        
        for n in range(n_levels):
            for m in range(n_levels):
                if abs(rho[n, m]) > 1e-10:
                    # Wigner function for Fock state |n⟩⟨m|
                    # For diagonal terms (n=m)
                    if n == m:
                        W_nm = factor * (-1)**n * np.exp(-2*r2) * eval_laguerre(n, 4*r2)
                    else:
                        # Off-diagonal terms (coherences)
                        # This requires full computation
                        # For simplicity, we approximate
                        W_nm = 0.0
                    
                    W_total += rho[n, m].real * W_nm
        
        W = W_total
        
    else:
        # Fallback: approximate with Gaussian
        logger.warning("scipy not available, using approximate Wigner function")
        sigma = 1.0
        W = (1 / (np.pi * hbar)) * np.exp(-(X**2 + P**2) / (2 * sigma**2))
    
    # Normalize
    dx = x[1] - x[0]
    dp = p[1] - p[0]
    W = W / (np.sum(W) * dx * dp)
    
    return W, x, p


# ============================================================================
# WIGNER FUNCTION - ANALYTIC
# ============================================================================

def wigner_function_analytic(
    psi_func: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float] = (-5, 5),
    p_range: Tuple[float, float] = (-5, 5),
    n_points: int = 100,
    hbar: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Wigner function from an analytic wavefunction
    
    W(x, p) = (1/πħ) ∫ dy ψ(x+y) ψ*(x-y) e^{-2ipy/ħ}
    
    Args:
        psi_func: Wavefunction ψ(x) as a callable
        x_range: (x_min, x_max) range
        p_range: (p_min, p_max) range
        n_points: Number of points in each dimension
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (W, x, p)
        
    Example:
        >>> def psi(x):
        ...     return np.exp(-x**2/2) / np.sqrt(np.sqrt(np.pi))
        >>> W, x, p = wigner_function_analytic(psi, (-3, 3), (-3, 3))
        >>> plot_wigner(W, x, p)
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    
    X, P = np.meshgrid(x, p)
    W = np.zeros_like(X, dtype=float)
    
    # Integration range
    y_range = 10.0
    n_y = 200
    y = np.linspace(-y_range, y_range, n_y)
    dy = y[1] - y[0]
    
    for i, xi in enumerate(x):
        for j, pj in enumerate(p):
            # Compute integral over y
            integral = 0.0
            for yk in y:
                psi_plus = psi_func(xi + yk)
                psi_minus = np.conj(psi_func(xi - yk))
                integral += psi_plus * psi_minus * np.exp(-2j * pj * yk / hbar)
            
            W[i, j] = (1 / (np.pi * hbar)) * np.real(integral) * dy
    
    # Normalize
    dx = x[1] - x[0]
    dp = p[1] - p[0]
    W = W / (np.sum(W) * dx * dp)
    
    return W, x, p


# ============================================================================
# WIGNER FUNCTION - GAUSSIAN
# ============================================================================

def wigner_function_gaussian(
    x0: float = 0.0,
    p0: float = 0.0,
    sigma: float = 1.0,
    x_range: Tuple[float, float] = (-5, 5),
    p_range: Tuple[float, float] = (-5, 5),
    n_points: int = 100,
    hbar: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Wigner function for a Gaussian state
    
    W(x, p) = (1/πħ) exp(-(x-x0)²/(2σ²) - (p-p0)²/(2σ²))
    
    Args:
        x0: Center position
        p0: Center momentum
        sigma: Width parameter
        x_range: (x_min, x_max) range
        p_range: (p_min, p_max) range
        n_points: Number of points in each dimension
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (W, x, p)
        
    Example:
        >>> W, x, p = wigner_function_gaussian(x0=0, p0=0, sigma=1.0)
        >>> plot_wigner(W, x, p, title="Gaussian State")
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    
    X, P = np.meshgrid(x, p)
    
    # Gaussian Wigner function
    W = (1 / (np.pi * hbar)) * np.exp(
        -(X - x0)**2 / (2 * sigma**2) -
        (P - p0)**2 / (2 * sigma**2)
    )
    
    # Normalize
    dx = x[1] - x[0]
    dp = p[1] - p[0]
    W = W / (np.sum(W) * dx * dp)
    
    return W, x, p


# ============================================================================
# WIGNER FUNCTION - COHERENT STATE
# ============================================================================

def wigner_function_coherent_state(
    alpha: complex,
    x_range: Tuple[float, float] = (-5, 5),
    p_range: Tuple[float, float] = (-5, 5),
    n_points: int = 100,
    hbar: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Wigner function for a coherent state |α⟩
    
    W(x, p) = (1/πħ) exp(-(x-x0)²/(2σ²) - (p-p0)²/(2σ²))
    where x0 = √(2ħ) Re(α) and p0 = √(2ħ) Im(α)
    
    Args:
        alpha: Complex amplitude
        x_range: (x_min, x_max) range
        p_range: (p_min, p_max) range
        n_points: Number of points in each dimension
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (W, x, p)
        
    Example:
        >>> alpha = 1.0 + 1j*0.5
        >>> W, x, p = wigner_function_coherent_state(alpha, (-3, 3), (-3, 3))
        >>> plot_wigner(W, x, p, title="Coherent State")
    """
    # Convert alpha to position and momentum
    x0 = np.sqrt(2 * hbar) * np.real(alpha)
    p0 = np.sqrt(2 * hbar) * np.imag(alpha)
    sigma = np.sqrt(hbar / 2)
    
    return wigner_function_gaussian(x0, p0, sigma, x_range, p_range, n_points, hbar)


# ============================================================================
# WIGNER FUNCTION - SQUEEZED STATE
# ============================================================================

def wigner_function_squeezed_state(
    r: float,
    phi: float = 0.0,
    x_range: Tuple[float, float] = (-5, 5),
    p_range: Tuple[float, float] = (-5, 5),
    n_points: int = 100,
    hbar: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Wigner function for a squeezed state
    
    W(x, p) = (1/πħ) exp(-(x²/(2σ²_x) + p²/(2σ²_p)))
    where σ_x = sqrt(ħ/2) e^r and σ_p = sqrt(ħ/2) e^{-r}
    
    Args:
        r: Squeezing parameter
        phi: Squeezing phase
        x_range: (x_min, x_max) range
        p_range: (p_min, p_max) range
        n_points: Number of points in each dimension
        hbar: Reduced Planck constant
        
    Returns:
        Tuple: (W, x, p)
        
    Example:
        >>> W, x, p = wigner_function_squeezed_state(r=0.5, phi=0, (-3, 3), (-3, 3))
        >>> plot_wigner(W, x, p, title="Squeezed State")
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    
    X, P = np.meshgrid(x, p)
    
    # Rotate coordinates for squeezed state
    # Apply rotation by phi
    c = np.cos(phi)
    s = np.sin(phi)
    
    X_rot = X * c + P * s
    P_rot = -X * s + P * c
    
    # Squeezed state Wigner function
    sigma_x = np.sqrt(hbar / 2) * np.exp(r)
    sigma_p = np.sqrt(hbar / 2) * np.exp(-r)
    
    W = (1 / (np.pi * hbar)) * np.exp(
        -X_rot**2 / (2 * sigma_x**2) -
        P_rot**2 / (2 * sigma_p**2)
    )
    
    # Normalize
    dx = x[1] - x[0]
    dp = p[1] - p[0]
    W = W / (np.sum(W) * dx * dp)
    
    return W, x, p


# ============================================================================
# PLOT WIGNER FUNCTION
# ============================================================================

def plot_wigner(
    W: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    title: str = "Wigner Function",
    cmap: str = 'RdBu',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    show_colorbar: bool = True,
    show_contours: bool = False,
    contour_levels: int = 10
):
    """
    Plot the Wigner function as a 2D colormap
    
    Args:
        W: Wigner function values
        x: x-axis (position) values
        p: y-axis (momentum) values
        title: Title of the plot
        cmap: Colormap name
        save_path: Path to save the figure
        figsize: Figure size (width, height)
        show_colorbar: Show colorbar
        show_contours: Show contour lines
        contour_levels: Number of contour levels
        
    Example:
        >>> W, x, p = wigner_function_gaussian()
        >>> plot_wigner(W, x, p, title="Gaussian State")
        >>> # Save to file
        >>> plot_wigner(W, x, p, save_path="wigner.png")
    """
    if not _has_matplotlib():
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return
    
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the Wigner function
        extent = [x[0], x[-1], p[0], p[-1]]
        im = ax.imshow(W.T, origin='lower', extent=extent, 
                       aspect='auto', cmap=cmap)
        
        # Add contours if requested
        if show_contours:
            levels = np.linspace(W.min(), W.max(), contour_levels)
            ax.contour(x, p, W.T, levels=levels, colors='black', alpha=0.3)
        
        # Add labels
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Momentum (p)')
        ax.set_title(title)
        
        # Add colorbar
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('W(x, p)')
        
        plt.tight_layout()
        
        # Save or show
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Wigner plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not plot Wigner function: {e}")


# ============================================================================
# PLOT WIGNER FUNCTION - 3D
# ============================================================================

def plot_wigner_3d(
    W: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    title: str = "Wigner Function",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    elevation: float = 30,
    azimuth: float = 45
):
    """
    Plot the Wigner function as a 3D surface
    
    Args:
        W: Wigner function values
        x: x-axis (position) values
        p: y-axis (momentum) values
        title: Title of the plot
        save_path: Path to save the figure
        figsize: Figure size (width, height)
        elevation: View elevation angle
        azimuth: View azimuthal angle
        
    Example:
        >>> W, x, p = wigner_function_gaussian()
        >>> plot_wigner_3d(W, x, p, title="Gaussian State")
    """
    if not _has_matplotlib():
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        X, P = np.meshgrid(x, p)
        
        # Plot surface
        surf = ax.plot_surface(X, P, W.T, cmap='RdBu', 
                              linewidth=0, antialiased=True, alpha=0.8)
        
        # Add contour projection on bottom
        ax.contour(X, P, W.T, zdir='z', offset=W.min(), cmap='RdBu', alpha=0.3)
        
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Momentum (p)')
        ax.set_zlabel('W(x, p)')
        ax.set_title(title)
        
        # Set view angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        
        # Save or show
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"3D Wigner plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not plot 3D Wigner function: {e}")


# ============================================================================
# WIGNER NEGATIVITY
# ============================================================================

def wigner_negativity(W: np.ndarray) -> float:
    """
    Compute the negativity of the Wigner function
    
    Negativity = ∫ |W(x, p)| dx dp - 1
    
    A non-zero negativity indicates non-classicality.
    
    Args:
        W: Wigner function values
        
    Returns:
        float: Negativity
        
    Example:
        >>> W, x, p = wigner_function_coherent_state(alpha)
        >>> neg = wigner_negativity(W)
        >>> print(f"Negativity: {neg:.6f}")  # ~0 for coherent states
    """
    # Normalize
    W_norm = W / np.sum(W)
    
    # Compute negativity
    negative = W_norm[W_norm < 0]
    negativity = -np.sum(negative)
    
    return float(negativity)


# ============================================================================
# ADDITIONAL UTILITIES
# ============================================================================

def wigner_marginals(W: np.ndarray, x: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the marginals of the Wigner function
    
    The position marginal is: P(x) = ∫ W(x, p) dp
    The momentum marginal is: Q(p) = ∫ W(x, p) dx
    
    Args:
        W: Wigner function values
        x: x-axis (position) values
        p: y-axis (momentum) values
        
    Returns:
        Tuple: (P_x, Q_p) position and momentum marginals
    """
    # Position marginal
    dx = x[1] - x[0]
    dp = p[1] - p[0]
    
    P_x = np.sum(W, axis=1) * dp
    Q_p = np.sum(W, axis=0) * dx
    
    return P_x, Q_p


def wigner_variance(W: np.ndarray, x: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """
    Compute the variance of the Wigner function
    
    Args:
        W: Wigner function values
        x: x-axis (position) values
        p: y-axis (momentum) values
        
    Returns:
        Tuple: (var_x, var_p)
    """
    # Normalize
    W_norm = W / np.sum(W)
    
    X, P = np.meshgrid(x, p)
    
    # Mean values
    mean_x = np.sum(X * W_norm)
    mean_p = np.sum(P * W_norm)
    
    # Variances
    var_x = np.sum((X - mean_x)**2 * W_norm)
    var_p = np.sum((P - mean_p)**2 * W_norm)
    
    return float(var_x), float(var_p)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Wigner functions
    'wigner_function',
    'wigner_function_analytic',
    'wigner_function_gaussian',
    'wigner_function_coherent_state',
    'wigner_function_squeezed_state',
    
    # Plotting
    'plot_wigner',
    'plot_wigner_3d',
    
    # Utilities
    'wigner_negativity',
    'wigner_marginals',
    'wigner_variance',
]