

# Contributing to PSIQIT

Thank you for your interest in contributing to PSIQIT! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check the [existing issues](https://github.com/psiqit/psiqit/issues) to avoid duplicates.

When creating a bug report, please include:
- A clear and descriptive title.
- Steps to reproduce the issue.
- Expected vs. actual behavior.
- Python version and Operating System.
- A minimal code example that demonstrates the problem.

### Suggesting Features

Feature requests are welcome! Please open an issue with:
- A clear description of the feature.
- The use case or motivation behind it.
- An example of how it would be used (if applicable).

### Pull Requests

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes.
4. Add or update tests as needed.
5. Ensure all tests pass: `pytest tests/`
6. Follow the code style guidelines.
7. Commit your changes: `git commit -m 'feat: add amazing feature'`
8. Push to your fork: `git push origin feature/amazing-feature`
9. Open a Pull Request.

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/psiqit.git
   cd psiqit
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

- Follow PEP 8 guidelines.
- Use type hints for function signatures.
- Maximum line length: 100 characters.
- Use Black for code formatting: `black .`
- Use descriptive variable names.
- Add docstrings to all public functions and classes.

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is negative
    
    Example:
        >>> example_function(5, "test")
        True
    """
    pass
```

## Testing

- Write tests for all new features.
- Maintain test coverage above 90%.
- Run tests before submitting a PR: `pytest tests/ -v --cov=psiqit`
- Include both unit tests and integration tests where appropriate.

## Documentation

- Update docstrings for any changed functions.
- Add examples to demonstrate new features.
- Update relevant documentation files in the `docs/` directory.
- Ensure all code examples are tested and working.

## Commit Messages

Use clear and descriptive commit messages following the Conventional Commits specification:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, missing semi-colons, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

**Example:**

```text
feat: Add support for multi-controlled gates

- Implemented CCZ gate decomposition
- Added tests for 3-qubit controlled operations
- Updated documentation with examples
```

## Review Process

- All PRs require at least one review before merging.
- CI tests must pass successfully.
- Code coverage must not decrease.
- Documentation must be updated accordingly.

## Questions?

Feel free to open an issue or contact the maintainer at [psiqitofficial@protonmail.com](mailto:psiqitofficial@protonmail.com).

Thank you for contributing to PSIQIT!

