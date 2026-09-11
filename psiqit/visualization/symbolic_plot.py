# psiqit/visualization/symbolic_plot.py

"""
Symbolic Plot Module
Render mathematical equations and symbols in LaTeX
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
import sys
import os

# Add parent directory to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..utils.logger import logger
except ImportError:
    # Fallback logger if running standalone
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

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


# ============================================================================
# RENDER EQUATION
# ============================================================================

def render_equation(
    latex: str,
    output_file: Optional[str] = None,
    fontsize: int = 16,
    dpi: int = 150,
    figsize: Tuple[float, float] = (8, 2)
) -> str:
    """
    Render a LaTeX equation as an image or return as string
    
    Args:
        latex: LaTeX equation string
        output_file: Path to save the image (if None, returns LaTeX string)
        fontsize: Font size for the equation
        dpi: Image resolution
        figsize: Figure size (width, height)
        
    Returns:
        str: LaTeX string or path to saved image
        
    Example:
        >>> latex = r"E = mc^2"
        >>> render_equation(latex, output_file="einstein.png")
        >>> # Or display in notebook
        >>> from IPython.display import display, Latex
        >>> display(Latex(render_equation(latex)))
    """
    if output_file is not None and _has_matplotlib():
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"${latex}$", 
                   fontsize=fontsize, ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight', transparent=True)
            plt.close()
            
            logger.info(f"Equation rendered to {output_file}")
            return output_file
            
        except Exception as e:
            logger.warning(f"Could not render equation: {e}")
    
    return f"${latex}$"


# ============================================================================
# SCHRODINGER EQUATION
# ============================================================================

def schrodinger_equation(time_dependent: bool = True) -> str:
    """
    Get the Schrodinger equation in LaTeX format
    
    Args:
        time_dependent: If True, returns time-dependent Schrodinger equation
                        If False, returns time-independent Schrodinger equation
        
    Returns:
        str: LaTeX equation string
        
    Example:
        >>> print(schrodinger_equation())
        i\\hbar\\frac{\\partial}{\\partial t}\\psi = \\hat{H}\\psi
        >>> print(schrodinger_equation(time_dependent=False))
        \\hat{H}\\psi = E\\psi
    """
    if time_dependent:
        return r"i\hbar\frac{\partial}{\partial t}\psi = \hat{H}\psi"
    else:
        return r"\hat{H}\psi = E\psi"


# ============================================================================
# DIRAC NOTATION
# ============================================================================

def dirac_notation() -> Dict[str, str]:
    """
    Get common Dirac notation symbols in LaTeX
    
    Returns:
        Dict: Mapping of notation to LaTeX code
        
    Example:
        >>> dirac = dirac_notation()
        >>> print(dirac['ket_zero'])  # |0\\rangle
        >>> print(dirac['inner_product'])  # \\langle \\phi | \\psi \\rangle
    """
    return {
        'ket_zero': r'|0\rangle',
        'ket_one': r'|1\rangle',
        'ket_plus': r'|+\rangle',
        'ket_minus': r'|-\rangle',
        'ket_psi': r'|\psi\rangle',
        'ket_phi': r'|\phi\rangle',
        'bra_zero': r'\langle 0|',
        'bra_one': r'\langle 1|',
        'bra_plus': r'\langle +|',
        'bra_minus': r'\langle -|',
        'bra_psi': r'\langle \psi|',
        'bra_phi': r'\langle \phi|',
        'inner_product': r'\langle \phi | \psi \rangle',
        'outer_product': r'|\psi\rangle\langle \psi|',
        'projector': r'|\psi\rangle\langle \psi|',
        'trace': r'\text{Tr}',
        'density_matrix': r'\rho = |\psi\rangle\langle \psi|',
    }


# ============================================================================
# PAULI MATRICES
# ============================================================================

def pauli_matrices() -> Dict[str, str]:
    """
    Get Pauli matrices in LaTeX format
    
    Returns:
        Dict: Pauli matrices as LaTeX strings
        
    Example:
        >>> pauli = pauli_matrices()
        >>> print(pauli['X'])
        \\sigma_x = \\begin{pmatrix} 0 & 1 \\\\ 1 & 0 \\end{pmatrix}
    """
    return {
        'I': r'I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}',
        'X': r'\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}',
        'Y': r'\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}',
        'Z': r'\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}',
        'sigma_x': r'\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}',
        'sigma_y': r'\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}',
        'sigma_z': r'\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}',
        'commutation': r'[\sigma_i, \sigma_j] = 2i\epsilon_{ijk}\sigma_k',
        'anticommutation': r'\{\sigma_i, \sigma_j\} = 2\delta_{ij}I',
    }


# ============================================================================
# BLOCH SPHERE EQUATION
# ============================================================================

def bloch_sphere_equation(theta: Optional[float] = None, phi: Optional[float] = None) -> str:
    """
    Get the Bloch sphere equation in LaTeX format
    
    Args:
        theta: Polar angle (if provided, includes specific state)
        phi: Azimuthal angle (if provided, includes specific state)
        
    Returns:
        str: LaTeX equation string
        
    Example:
        >>> print(bloch_sphere_equation())
        |\\psi\\rangle = \\cos(\\theta/2)|0\\rangle + e^{i\\phi}\\sin(\\theta/2)|1\\rangle
        >>> print(bloch_sphere_equation(theta=np.pi/4, phi=np.pi/2))
        |\\psi\\rangle = \\cos(\\pi/8)|0\\rangle + e^{i\\pi/2}\\sin(\\pi/8)|1\\rangle
    """
    if theta is not None and phi is not None:
        theta_str = f"{theta:.2f}" if isinstance(theta, float) else str(theta)
        phi_str = f"{phi:.2f}" if isinstance(phi, float) else str(phi)
        return rf"|\psi\rangle = \cos({theta_str}/2)|0\rangle + e^{{i{phi_str}}}\sin({theta_str}/2)|1\rangle"
    elif theta is not None:
        theta_str = f"{theta:.2f}" if isinstance(theta, float) else str(theta)
        return rf"|\psi\rangle = \cos({theta_str}/2)|0\rangle + e^{{i\phi}}\sin({theta_str}/2)|1\rangle"
    else:
        return r"|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle"


# ============================================================================
# HEISENBERG UNCERTAINTY
# ============================================================================

def heisenberg_uncertainty() -> str:
    """
    Get the Heisenberg uncertainty principle in LaTeX format
    
    Returns:
        str: LaTeX equation string
        
    Example:
        >>> print(heisenberg_uncertainty())
        \\Delta x \\Delta p \\geq \\frac{\\hbar}{2}
    """
    return r"\Delta x \Delta p \geq \frac{\hbar}{2}"


# ============================================================================
# COMMUTATION RELATION
# ============================================================================

def commutation_relation() -> str:
    """
    Get the canonical commutation relation in LaTeX format
    
    Returns:
        str: LaTeX equation string
        
    Example:
        >>> print(commutation_relation())
        [x, p] = i\\hbar
    """
    return r"[x, p] = i\hbar"


# ============================================================================
# MAXWELL'S EQUATIONS
# ============================================================================

def maxwell_equations() -> List[str]:
    """
    Get Maxwell's equations in LaTeX format
    
    Returns:
        List[str]: List of LaTeX equation strings
        
    Example:
        >>> for eq in maxwell_equations():
        ...     print(eq)
        \\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\varepsilon_0}
        \\nabla \\cdot \\mathbf{B} = 0
        \\nabla \\times \\mathbf{E} = -\\frac{\\partial \\mathbf{B}}{\\partial t}
        \\nabla \\times \\mathbf{B} = \\mu_0 \\mathbf{J} + \\mu_0 \\varepsilon_0 \\frac{\\partial \\mathbf{E}}{\\partial t}
    """
    return [
        r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}",
        r"\nabla \cdot \mathbf{B} = 0",
        r"\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}",
        r"\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}",
    ]


# ============================================================================
# ADDITIONAL QUANTUM EQUATIONS
# ============================================================================

def quantum_equations() -> Dict[str, str]:
    """
    Get common quantum mechanics equations in LaTeX format
    
    Returns:
        Dict: Mapping of equation names to LaTeX strings
        
    Example:
        >>> eqs = quantum_equations()
        >>> print(eqs['schrodinger'])
        i\\hbar\\frac{\\partial}{\\partial t}\\psi = \\hat{H}\\psi
    """
    return {
        'schrodinger': r"i\hbar\frac{\partial}{\partial t}\psi = \hat{H}\psi",
        'schrodinger_stationary': r"\hat{H}\psi = E\psi",
        'heisenberg': r"\frac{d\hat{A}}{dt} = \frac{i}{\hbar}[\hat{H}, \hat{A}] + \frac{\partial \hat{A}}{\partial t}",
        'lindblad': r"\frac{d\rho}{dt} = -\frac{i}{\hbar}[\hat{H}, \rho] + \sum_k \gamma_k (L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\})",
        'density_matrix': r"\rho = \sum_i p_i |\psi_i\rangle\langle \psi_i|",
        'fidelity': r"F(\rho, \sigma) = \text{Tr}(\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})^2",
        'entanglement_entropy': r"S(\rho_A) = -\text{Tr}(\rho_A \log \rho_A)",
        'concurrence': r"C(\rho) = \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4)",
        'negativity': r"\mathcal{N}(\rho) = \frac{\|\rho^{T_A}\|_1 - 1}{2}",
        'uncertainty': r"\Delta x \Delta p \geq \frac{\hbar}{2}",
        'commutation': r"[x, p] = i\hbar",
        'bloch': r"|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle",
        'bell': r"|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}",
        'ghz': r"|\text{GHZ}\rangle = \frac{|00\ldots 0\rangle + |11\ldots 1\rangle}{\sqrt{2}}",
        'w': r"|W\rangle = \frac{|100\ldots 0\rangle + |010\ldots 0\rangle + \ldots + |000\ldots 1\rangle}{\sqrt{n}}",
    }


# ============================================================================
# RENDER EQUATION AS IMAGE
# ============================================================================

def render_equation_as_image(
    latex: str,
    output_path: str,
    fontsize: int = 20,
    dpi: int = 200,
    figsize: Tuple[float, float] = (10, 2),
    bg_color: str = 'white',
    text_color: str = 'black'
) -> bool:
    """
    Render a LaTeX equation as an image file
    
    Args:
        latex: LaTeX equation string
        output_path: Path to save the image
        fontsize: Font size
        dpi: Image resolution
        figsize: Figure size (width, height)
        bg_color: Background color
        text_color: Text color
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> success = render_equation_as_image(
        ...     r"E = mc^2",
        ...     "einstein.png",
        ...     fontsize=24
        ... )
    """
    if not _has_matplotlib():
        logger.warning("matplotlib not available. Install with: pip install matplotlib")
        return False
    
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        ax.text(0.5, 0.5, f"${latex}$", 
               fontsize=fontsize, ha='center', va='center',
               color=text_color)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor=bg_color, edgecolor='none')
        plt.close()
        
        logger.info(f"Equation rendered to {output_path}")
        return True
        
    except Exception as e:
        logger.warning(f"Could not render equation: {e}")
        return False


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'render_equation',
    'schrodinger_equation',
    'dirac_notation',
    'pauli_matrices',
    'bloch_sphere_equation',
    'heisenberg_uncertainty',
    'commutation_relation',
    'maxwell_equations',
    'quantum_equations',
    'render_equation_as_image',
]