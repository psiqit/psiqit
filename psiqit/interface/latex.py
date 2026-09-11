# psiqit/interface/latex.py
r"""
LaTeX Output Module
Generate LaTeX code for matrices, states, circuits, and reports
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from ..quantum.state import Ket
from ..quantum.operator import Operator
from ..circuits.circuit import QuantumCircuit
from ..utils.logger import logger


# ============================================================================
# MATRIX TO LATEX
# ============================================================================

def matrix_to_latex(
    matrix: Union[List[List], np.ndarray, Operator],
    name: str = "",
    precision: int = 4,
    as_array: bool = False
) -> str:
    """
    Convert a matrix to LaTeX format
    
    Args:
        matrix: Matrix data (list, numpy array, or Operator)
        name: Name of the matrix (optional)
        precision: Number of decimal places
        as_array: Use array environment instead of pmatrix
        
    Returns:
        str: LaTeX code
        
    Example:
        >>> from psiqit.quantum import pauli_x
        >>> print(matrix_to_latex(pauli_x(), name="X"))
        X = \begin{pmatrix}
        0 & 1 \\
        1 & 0
        \end{pmatrix}
    """
    if isinstance(matrix, Operator):
        data = matrix.data
        matrix_name = matrix.name if matrix.name else name
    else:
        data = np.array(matrix, dtype=complex)
        matrix_name = name
    
    # Determine format
    rows, cols = data.shape
    env = "array" if as_array else "pmatrix"
    
    # Build matrix string
    lines = []
    
    if matrix_name:
        lines.append(f"{matrix_name} = \\begin{{{env}}}")
    else:
        lines.append(f"\\begin{{{env}}}")
    
    for i in range(rows):
        row_elements = []
        for j in range(cols):
            val = data[i, j]
            if abs(val.imag) < 1e-10:
                # Real number
                row_elements.append(f"{val.real:.{precision}f}")
            elif abs(val.real) < 1e-10:
                # Pure imaginary
                row_elements.append(f"{val.imag:.{precision}f}i")
            else:
                # Complex number
                row_elements.append(f"{val.real:.{precision}f} + {val.imag:.{precision}f}i")
        lines.append(" & ".join(row_elements) + " \\\\")
    
    lines.append(f"\\end{{{env}}}")
    
    return "\n".join(lines)


# ============================================================================
# STATE TO LATEX
# ============================================================================

def state_to_latex(
    state: Union[List, np.ndarray, Ket],
    name: str = "\\psi",
    precision: int = 4
) -> str:
    """
    Convert a quantum state to LaTeX format
    
    Args:
        state: State vector (list, numpy array, or Ket)
        name: Name of the state (e.g., "\\psi", "\\phi")
        precision: Number of decimal places
        
    Returns:
        str: LaTeX code
        
    Example:
        >>> from psiqit.quantum import Ket
        >>> state = Ket([1/np.sqrt(2), 1/np.sqrt(2)])
        >>> print(state_to_latex(state))
        |\psi\rangle = \frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle
    """
    if isinstance(state, Ket):
        data = state.data
    else:
        data = np.array(state, dtype=complex)
    
    # Normalize
    norm = np.linalg.norm(data)
    if norm > 0:
        data = data / norm
    
    # Build state string
    terms = []
    dim = len(data)
    bit_length = int(np.ceil(np.log2(dim))) if dim > 1 else 1
    
    for i, amp in enumerate(data):
        if abs(amp) < 1e-10:
            continue
        
        # Format amplitude
        if abs(amp.imag) < 1e-10:
            amp_str = f"{amp.real:.{precision}f}"
        elif abs(amp.real) < 1e-10:
            amp_str = f"{amp.imag:.{precision}f}i"
        else:
            amp_str = f"({amp.real:.{precision}f} + {amp.imag:.{precision}f}i)"
        
        # Format basis state
        if dim == 2:
            basis_str = f"|{i}\\rangle"
        else:
            basis_str = f"|{i:0{bit_length}b}\\rangle"
        
        # Add term
        if i == 0:
            terms.append(f"{amp_str}{basis_str}")
        else:
            if amp.real < 0 or (abs(amp.real) < 1e-10 and amp.imag < 0):
                terms.append(f" - {abs(amp.real):.{precision}f}{basis_str}" if abs(amp.imag) < 1e-10 else f" - {amp_str}{basis_str}")
            else:
                terms.append(f" + {amp_str}{basis_str}")
    
    if not terms:
        return f"|{name}\\rangle = 0"
    
    state_str = "".join(terms)
    
    return f"|{name}\\rangle = {state_str}"


# ============================================================================
# CIRCUIT TO LATEX
# ============================================================================

def circuit_to_latex(
    circuit: QuantumCircuit,
    style: str = 'quantikz',
    wire_labels: Optional[List[str]] = None
) -> str:
    """
    Convert a quantum circuit to LaTeX using Quantikz package
    
    Args:
        circuit: QuantumCircuit object
        style: 'quantikz' or 'qcircuit'
        wire_labels: List of wire labels (default: q0, q1, ...)
        
    Returns:
        str: LaTeX code
        
    Example:
        >>> from psiqit.circuits import QuantumCircuit
        >>> circ = QuantumCircuit(2)
        >>> circ.h(0).cx(0, 1)
        >>> print(circuit_to_latex(circ))
    """
    if style == 'quantikz':
        return _circuit_to_quantikz(circuit, wire_labels)
    elif style == 'qcircuit':
        return _circuit_to_qcircuit(circuit, wire_labels)
    else:
        raise ValueError(f"Unknown style: {style}. Use 'quantikz' or 'qcircuit'")


def _circuit_to_quantikz(
    circuit: QuantumCircuit,
    wire_labels: Optional[List[str]] = None
) -> str:
    """
    Convert circuit to Quantikz format
    """
    n_qubits = circuit.n_qubits
    
    # Default wire labels
    if wire_labels is None:
        wire_labels = [f"q_{i}" for i in range(n_qubits)]
    
    # Build the circuit
    lines = []
    lines.append("\\begin{quantikz}")
    
    # Add wires
    for i in range(n_qubits):
        line = f"\\lstick{{{wire_labels[i]}}} & "
        
        # Add gates
        gates = circuit.get_gates()
        gate_count = 0
        for gate in gates:
            name = gate['name']
            qubits = gate['qubits']
            params = gate.get('params', [])
            
            if i in qubits:
                if name in ['H', 'X', 'Y', 'Z', 'S', 'T']:
                    line += f"\\gate{{{name}}} & "
                elif name in ['RX', 'RY', 'RZ']:
                    theta = params[0] if params else 0
                    line += f"\\gate{{{name}({theta:.2f})}} & "
                elif name == 'CNOT':
                    if qubits[0] == i:
                        line += "\\ctrl{1} & "
                    elif qubits[1] == i:
                        line += "\\targ{} & "
                elif name == 'CZ':
                    if qubits[0] == i:
                        line += "\\ctrl{1} & "
                    elif qubits[1] == i:
                        line += "\\gate{Z} & "
                elif name == 'SWAP':
                    if qubits[0] == i:
                        line += "\\swap{1} & "
                    elif qubits[1] == i:
                        line += "\\swap{-1} & "
                elif name == 'TOFFOLI':
                    # Simplified representation
                    line += "\\gate{CCX} & "
                else:
                    line += f"\\gate{{{name}}} & "
                
                gate_count += 1
            else:
                # Empty wire
                line += "\\qw & "
        
        # Remove trailing ampersand and add newline
        if line.endswith(" & "):
            line = line[:-3]
        lines.append(line + " \\\\")
    
    lines.append("\\end{quantikz}")
    
    return "\n".join(lines)


def _circuit_to_qcircuit(
    circuit: QuantumCircuit,
    wire_labels: Optional[List[str]] = None
) -> str:
    """
    Convert circuit to QCircuit format
    """
    # This is a simplified version
    # QCircuit is an older package, but still useful
    n_qubits = circuit.n_qubits
    
    if wire_labels is None:
        wire_labels = [f"q_{i}" for i in range(n_qubits)]
    
    lines = []
    lines.append("\\begin{figure}")
    lines.append("\\Qcircuit @C=1em @R=.7em {")
    
    # Build each wire
    for i in range(n_qubits):
        line = f"& \\lstick{{{wire_labels[i]}}} "
        gates = circuit.get_gates()
        
        for gate in gates:
            name = gate['name']
            qubits = gate['qubits']
            
            if i in qubits:
                if name == 'H':
                    line += "& \\gate{H} "
                elif name == 'X':
                    line += "& \\gate{X} "
                elif name == 'Y':
                    line += "& \\gate{Y} "
                elif name == 'Z':
                    line += "& \\gate{Z} "
                elif name == 'CNOT':
                    if qubits[0] == i:
                        line += "& \\ctrl{1} "
                    elif qubits[1] == i:
                        line += "& \\targ "
                else:
                    line += f"& \\gate{{{name}}} "
            else:
                line += "& \\qw "
        
        line += "\\\\"
        lines.append(line)
    
    lines.append("}")
    lines.append("\\end{figure}")
    
    return "\n".join(lines)


# ============================================================================
# EQUATION TO LATEX
# ============================================================================

def equation_to_latex(
    expr: str,
    name: str = "",
    align: bool = False
) -> str:
    """
    Format a mathematical expression as LaTeX equation
    
    Args:
        expr: Mathematical expression (LaTeX string)
        name: Name of the equation (optional)
        align: Use align environment instead of equation
        
    Returns:
        str: LaTeX code
        
    Example:
        >>> print(equation_to_latex("E = mc^2", name="Einstein"))
        \\begin{equation}
        \\label{eq:Einstein}
        E = mc^2
        \\end{equation}
    """
    lines = []
    
    if align:
        lines.append("\\begin{align}")
        if name:
            lines.append(f"\\label{{eq:{name}}}")
        lines.append(expr)
        lines.append("\\end{align}")
    else:
        lines.append("\\begin{equation}")
        if name:
            lines.append(f"\\label{{eq:{name}}}")
        lines.append(expr)
        lines.append("\\end{equation}")
    
    return "\n".join(lines)


# ============================================================================
# TABLE TO LATEX
# ============================================================================

def table_to_latex(
    data: List[List[Any]],
    headers: Optional[List[str]] = None,
    caption: str = "",
    label: str = "",
    column_format: Optional[str] = None
) -> str:
    """
    Convert data to LaTeX table
    
    Args:
        data: 2D list of data
        headers: Column headers (optional)
        caption: Table caption
        label: Table label
        column_format: Column format string (e.g., "|c|c|c|")
        
    Returns:
        str: LaTeX code
        
    Example:
        >>> data = [[1, 2], [3, 4]]
        >>> headers = ['A', 'B']
        >>> print(table_to_latex(data, headers, caption="My Table"))
    """
    if not data:
        return "\\begin{tabular}{c}\nNo data\\end{tabular}"
    
    n_cols = len(data[0])
    
    # Determine column format
    if column_format is None:
        column_format = "c" * n_cols
        if headers:
            column_format = "|" + "|".join(["c"] * n_cols) + "|"
    
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{column_format}}}")
    lines.append("\\hline")
    
    # Headers
    if headers:
        header_line = " & ".join(headers) + " \\\\"
        lines.append(header_line)
        lines.append("\\hline")
    
    # Data
    for row in data:
        # Convert all elements to strings
        row_strs = [str(val) for val in row]
        row_line = " & ".join(row_strs) + " \\\\"
        lines.append(row_line)
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{tab:{label}}}")
    
    lines.append("\\end{table}")
    
    return "\n".join(lines)


# ============================================================================
# GENERATE REPORT
# ============================================================================

def generate_report(
    results: Dict[str, Any],
    title: str = "Quantum Report",
    author: str = "",
    date: str = ""
) -> str:
    """
    Generate a complete LaTeX report from results
    
    Args:
        results: Dictionary containing results
        title: Report title
        author: Author name
        date: Date (default: today)
        
    Returns:
        str: Full LaTeX document
        
    Example:
        >>> results = {
        ...     'circuit': {'n_qubits': 2, 'depth': 3},
        ...     'measurement': {'counts': {'00': 512, '11': 512}}
        ... }
        >>> print(generate_report(results, title="Bell State Report"))
    """
    from datetime import datetime
    
    if not date:
        date = datetime.now().strftime("%B %d, %Y")
    
    lines = []
    
    # Document header
    lines.append("\\documentclass[12pt]{article}")
    lines.append("\\usepackage{amsmath, amssymb, amsfonts}")
    lines.append("\\usepackage{graphicx}")
    lines.append("\\usepackage{geometry}")
    lines.append("\\geometry{margin=1in}")
    lines.append("")
    lines.append("\\begin{document}")
    lines.append("")
    
    # Title
    lines.append("\\title{" + title + "}")
    if author:
        lines.append("\\author{" + author + "}")
    lines.append("\\date{" + date + "}")
    lines.append("\\maketitle")
    lines.append("")
    
    # Abstract
    lines.append("\\begin{abstract}")
    lines.append("This report presents the results of quantum simulations ")
    lines.append("performed using PSIQIT (Python Scientific Quantum Information Toolkit).")
    lines.append("\\end{abstract}")
    lines.append("")
    
    # Sections
    lines.append("\\section{Introduction}")
    lines.append("The following results were obtained using PSIQIT.")
    lines.append("")
    
    # Results
    for key, value in results.items():
        lines.append(f"\\subsection{{{key.capitalize()}}}")
        
        if isinstance(value, dict):
            for subkey, subval in value.items():
                lines.append(f"\\textbf{{{subkey}}}: {subval}")
        else:
            lines.append(f"{value}")
        
        lines.append("")
    
    # Conclusion
    lines.append("\\section{Conclusion}")
    lines.append("The simulation completed successfully.")
    lines.append("")
    
    # End
    lines.append("\\end{document}")
    
    return "\n".join(lines)


# ============================================================================
# ADDITIONAL UTILITIES
# ============================================================================

def dirac_notation() -> Dict[str, str]:
    """
    Get common Dirac notation in LaTeX
    
    Returns:
        Dict: Mapping of notation to LaTeX code
        
    Example:
        >>> dirac = dirac_notation()
        >>> print(dirac['ket_zero'])  # |0\rangle
    """
    return {
        'ket_zero': r'|0\rangle',
        'ket_one': r'|1\rangle',
        'ket_plus': r'|+\rangle',
        'ket_minus': r'|-\rangle',
        'bra_zero': r'\langle 0|',
        'bra_one': r'\langle 1|',
        'bra_plus': r'\langle +|',
        'bra_minus': r'\langle -|',
        'bell_phi_plus': r'|\Phi^+\rangle',
        'bell_phi_minus': r'|\Phi^-\rangle',
        'bell_psi_plus': r'|\Psi^+\rangle',
        'bell_psi_minus': r'|\Psi^-\rangle',
    }


def pauli_matrices_latex() -> Dict[str, str]:
    """
    Get Pauli matrices in LaTeX format
    
    Returns:
        Dict: Pauli matrices as LaTeX strings
        
    Example:
        >>> pauli = pauli_matrices_latex()
        >>> print(pauli['X'])
        \sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
    """
    return {
        'I': r'I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}',
        'X': r'\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}',
        'Y': r'\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}',
        'Z': r'\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}',
    }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'matrix_to_latex',
    'state_to_latex',
    'circuit_to_latex',
    'equation_to_latex',
    'table_to_latex',
    'generate_report',
    'dirac_notation',
    'pauli_matrices_latex',
] 
