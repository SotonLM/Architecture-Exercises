# Setup Instructions

## 1. Install Git (if you don't have it)

- Mac: `git --version` (installs automatically if missing)

- Windows: Download from https://git-scm.com/

- Linux: `sudo apt-get install git`

## 2. Clone the Repository

```bash
# Navigate to where you want the project
cd ~/Documents  # or wherever you keep projects

# Clone the repo
git clone https://github.com/SotonLM/Architecture-Exercises.git

# Enter the directory and navigate to week3
cd Architecture-Exercises/week3
```

## 3. Install uv (Modern Python Package Manager)

**uv** is a fast Python package installer and resolver. Install it:

- **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

- **Mac/Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Or with pip:**
  ```bash
  pip install uv
  ```

Verify installation:
```bash
uv --version
```

## 4. Install Python (uv will handle this automatically)

uv can automatically install and manage Python versions. You can skip manual Python installation if using uv.

To check/manage Python versions:
```bash
uv python list    # List available Python versions
uv python install 3.11  # Install specific version (optional)
```

## 5. Create Virtual Environment and Install Dependencies

uv makes this easy - it will create a virtual environment and install dependencies automatically:

```bash
# Navigate to the week3 folder (if not already there)
cd Architecture-Exercises/week3

# Create virtual environment
uv venv

# Install dependencies
uv pip install transformers torch
```

This will:
- Create a virtual environment (`.venv` by default)
- Install transformers and torch
- Make everything ready to use

**Activate the virtual environment:**
```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

## 6. Test Installation
```bash
# Using uv (recommended)
uv run python -c "import transformers; print('Works!')"

# Or if virtual environment is activated
python -c "import transformers; print('Works!')"
```

## 7. Run Diagnostic Task
```bash
# Using uv (recommended - no need to activate venv)
uv run python diagnostic_task.py

# Or if virtual environment is activated
python diagnostic_task.py
```

## Troubleshooting
- **"uv: command not found"**: Make sure uv is installed and in your PATH. Restart terminal after installation.
- **"No module named transformers"**: Run `uv pip install transformers torch` again
- **Build errors with uv sync**: Use `uv venv` and `uv pip install` instead (this project is not a Python package)
- **Torch issues**: uv should handle torch installation automatically. If issues persist, try:
  ```bash
  uv pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- **Out of memory**: Use `distilgpt2` (smallest model)
- **Slow generation**: Normal on CPU, be patient
- **Python version issues**: Let uv manage Python: `uv python install 3.11`

## Getting Help
- Send a message to one of the leads!!
@Ali Almarzooq or @Sebastian Ene
