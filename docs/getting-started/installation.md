

# Installation Guide

This guide provides detailed instructions for installing PSIQIT in various environments and configurations.

## Prerequisites

Before installing PSIQIT, ensure you have the following installed on your system:

- **Python 3.8 or higher** ([Download Python](https://www.python.org/downloads/))
- **pip** (Python package installer, included with Python 3.4+)

!!! note "Check your Python version"
    You can verify your Python installation by running:
    ```bash
    python --version
    ```

---

## Standard Installation

For most users, the simplest and recommended way to install PSIQIT is via pip from PyPI:

```bash
pip install psiqit
```

This command installs the core library along with essential dependencies (`numpy`, `scipy`, `matplotlib`).

### Verify Installation

After installation, verify that PSIQIT is correctly installed and accessible:

```bash
python -c "import psiqit; print(psiqit.__version__)"
```

**Expected Output:**
```text
1.0.3
```

---

## Installation from Source (For Developers)

If you want to access the latest features, modify the source code, or contribute to PSIQIT, install directly from the GitHub repository:

```bash
# 1. Clone the repository
git clone https://github.com/psiqit/psiqit.git
cd psiqit

# 2. Install in editable mode with development dependencies
pip install -e ".[dev]"
```

!!! tip "Editable Mode"
    The `-e` flag installs PSIQIT in "editable" mode. This means any changes you make to the source code will be immediately reflected in your environment without needing to reinstall the package.

---

## Optional Dependencies

PSIQIT supports several optional feature sets. You can install them by specifying extras in brackets:

| Feature | Command | Includes |
| :--- | :--- | :--- |
| **Development** | `pip install "psiqit[dev]"` | `pytest`, `black`, `flake8`, `pre-commit` |
| **Documentation** | `pip install "psiqit[docs]"` | `mkdocs`, `mkdocs-material`, `mkdocstrings` |
| **All Features** | `pip install "psiqit[full]"` | All of the above combined |

---

## Virtual Environment (Highly Recommended)

To avoid dependency conflicts with other Python projects, we strongly recommend using a virtual environment.

=== "Using venv (Built-in)"
    ```bash
    # Create a virtual environment named 'psiqit-env'
    python -m venv psiqit-env

    # Activate it (Windows)
    psiqit-env\Scripts\activate

    # Activate it (macOS/Linux)
    source psiqit-env/bin/activate

    # Install PSIQIT
    pip install psiqit
    ```

=== "Using conda"
    ```bash
    # Create a conda environment
    conda create -n psiqit-env python=3.11

    # Activate it
    conda activate psiqit-env

    # Install PSIQIT
    pip install psiqit
    ```

---

## Troubleshooting

!!! warning "Permission Denied (Linux/macOS)"
    Avoid using `sudo pip install`. Instead, use the `--user` flag or a virtual environment:
    ```bash
    pip install --user psiqit
    ```

!!! bug "Dependency Conflicts"
    If you encounter dependency resolution errors, try upgrading pip first:
    ```bash
    python -m pip install --upgrade pip
    pip install psiqit
    ```

---

## Next Steps

Now that PSIQIT is installed, you are ready to write your first quantum circuit! Proceed to the **[Quick Start](quick-start.md)** guide.


