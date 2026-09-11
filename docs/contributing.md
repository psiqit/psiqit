
# Contributing to PSIQIT

Thank you for your interest in contributing to PSIQIT! This guide will help you get started with contributing to the project. :)

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear description of the bug
- Steps to reproduce it
- Expected vs actual behavior
- Your environment (Python version, OS, PSIQIT version)

### Suggesting Features

We welcome feature requests! Please open an issue and describe:
- The problem you're trying to solve
- Your proposed solution
- Alternative approaches you've considered

### Submitting Code

1. **Fork the repository** on GitHub
2. **Create a branch** for your feature: `git checkout -b feature/your-feature-name`
3. **Write tests** for your code (we use pytest)
4. **Follow the code style** (PEP 8, type hints, docstrings)
5. **Commit your changes**: `git commit -m "Add your feature"`
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Open a Pull Request** on GitHub

## Code Style Guidelines

### Python Style
- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints for all function signatures
- Write comprehensive docstrings (Google style)
- Keep functions focused and under 50 lines when possible

### Example Function
```python
def calculate_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Calculate the fidelity between two quantum states.
    
    Args:
        state1: First quantum state (statevector)
        state2: Second quantum state (statevector)
        
    Returns:
        Fidelity value between 0 and 1
        
    Example:
        >>> state1 = np.array([1, 0])
        >>> state2 = np.array([0, 1])
        >>> fidelity = calculate_fidelity(state1, state2)
        >>> print(fidelity)  # 0.0
    """
    return float(np.abs(np.vdot(state1, state2))**2)
```

### Documentation
- All public functions must have docstrings
- Include examples in docstrings when possible
- Update the relevant tutorial or API documentation

## Testing

Before submitting a PR, make sure all tests pass:
```bash
pytest tests/
```

## Questions?

Feel free to open an issue or contact the maintainers. We're here to help!

---

Thank you for helping make PSIQIT better! 🎉
