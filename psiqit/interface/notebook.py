# psiqit/interface/notebook.py
"""
Jupyter Notebook Tools Module
Display functions for quantum circuits, states, and visualizations in notebooks
"""

import sys
import numpy as np
from typing import Optional, List, Dict, Any, Union

# Check if running in IPython/Jupyter
_IN_IPYTHON = False
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        _IN_IPYTHON = True
        # Only execute magic command if in IPython
        ipython.run_line_magic('matplotlib', 'inline')
except (ImportError, NameError, AttributeError):
    pass

# Import IPython display functions (always available if IPython is installed)
try:
    from IPython.display import display, HTML, Markdown, Latex
except ImportError:
    # Fallback: define dummy display functions
    def display(*args, **kwargs):
        print(*args)
    
    def HTML(content):
        return content
    
    def Markdown(content):
        return content
    
    def Latex(content):
        return content

from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket
from ..quantum.operator import Operator
from ..utils.logger import logger
from ..version import __version__


# ============================================================================
# CHECK DEPENDENCIES
# ============================================================================

def _has_matplotlib() -> bool:
    """Check if matplotlib is available"""
    try:
        import matplotlib
        return True
    except ImportError:
        return False


def _has_plotly() -> bool:
    """Check if plotly is available"""
    try:
        import plotly
        return True
    except ImportError:
        return False


def _has_qiskit() -> bool:
    """Check if qiskit is available"""
    try:
        import qiskit
        return True
    except ImportError:
        return False


# ============================================================================
# INIT NOTEBOOK
# ============================================================================

def init_notebook():
    """
    Initialize Jupyter notebook for PSIQIT
    
    Sets up:
    - Matplotlib inline
    - Plotly for interactive visualizations
    - Qiskit (if available)
    - Pretty printing
    
    Example:
        >>> from psiqit.interface import init_notebook
        >>> init_notebook()
    """
    print(f"PSIQIT - Python Scientific Quantum Information Toolkit v{__version__}")
    print("=" * 60)
    
    # Display info
    if _IN_IPYTHON:
        try:
            display(HTML(f"""
            <div style="background: #f0f8ff; padding: 10px; border-radius: 5px;">
                <b>PSIQIT</b> - Python Scientific Quantum Information Toolkit v{__version__}
                <br>Ready for quantum computing!
            </div>
            """))
        except:
            print(f"PSIQIT v{__version__} loaded")
    
    # Check matplotlib
    if _has_matplotlib():
        print("✅ matplotlib loaded")
    else:
        print("⚠️ matplotlib not available (install: pip install matplotlib)")
    
    # Check plotly
    if _has_plotly():
        print("✅ plotly loaded")
    else:
        print("⚠️ plotly not available (install: pip install plotly)")
    
    # Check qiskit
    if _has_qiskit():
        print("✅ qiskit loaded")
    else:
        print("⚠️ qiskit not available (install: pip install qiskit)")
    
    print("=" * 60)
    print("Notebook initialized successfully!")


# ============================================================================
# DISPLAY CIRCUIT
# ============================================================================

def display_circuit(
    circuit: QuantumCircuit,
    style: str = 'text',
    title: str = "Quantum Circuit"
):
    """
    Display a quantum circuit in Jupyter notebook
    
    Args:
        circuit: QuantumCircuit object
        style: 'text', 'ascii', 'unicode', or 'matplotlib'
        title: Title to display above circuit
        
    Example:
        >>> from psiqit.circuits import QuantumCircuit
        >>> circ = QuantumCircuit(2)
        >>> circ.h(0).cx(0, 1)
        >>> display_circuit(circ)
    """
    try:
        from IPython.display import display, HTML, Markdown
    except ImportError:
        print(f"### {title}")
        print(f"**{circuit.n_qubits} qubits, {circuit.depth} depth, {len(circuit.get_gates())} gates**")
        print(circuit.draw(style='ascii'))
        return
    
    # Display title
    display(Markdown(f"### {title}"))
    display(Markdown(f"**{circuit.n_qubits} qubits, {circuit.depth} depth, {len(circuit.get_gates())} gates**"))
    
    # Draw circuit
    if style == 'text':
        draw_text = circuit.draw(style='ascii')
        display(HTML(f"<pre style='font-family: monospace;'>{draw_text}</pre>"))
    elif style == 'matplotlib':
        if _has_matplotlib():
            draw_text = circuit.draw(style='ascii')
            display(HTML(f"<pre style='font-family: monospace;'>{draw_text}</pre>"))
        else:
            draw_text = circuit.draw(style='ascii')
            display(HTML(f"<pre style='font-family: monospace;'>{draw_text}</pre>"))
    else:
        draw_text = circuit.draw(style='ascii')
        display(HTML(f"<pre style='font-family: monospace;'>{draw_text}</pre>"))


# ... بقیه توابع (display_state, display_histogram, display_bloch, display_matrix, display_notebook_version) به همین صورت باقی می‌مانند ...


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'init_notebook',
    'display_circuit',
    'display_state',
    'display_histogram',
    'display_bloch',
    'display_matrix',
    'display_notebook_version',
]