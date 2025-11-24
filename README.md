# Week 3: Diagnostic Task

This folder contains the diagnostic task for Week 3 of the SotonLM Architecture course.

## Structure

```
week3/
├── README.md                 # This file
├── SETUP.MD                  # Setup instructions
├── pyproject.toml            # Dependencies (for uv)
├── diagnostic_task.py        # Main task file
├── task.txt                  # Complete task description
├── submission_format.txt     # Submission template
├── level1/                   # Level 1 task details
│   └── README.md
├── level2/                   # Level 2 task details
│   └── README.md
├── level3/                   # Level 3 task details
│   └── README.md
└── level4/                   # Level 4 task details
    └── README.md
```

## Quick Start

1. **Install uv** (if not already installed)
   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Mac/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Setup Environment**
   ```bash
   # Create virtual environment
   uv venv
   
   # Install dependencies
   uv pip install transformers torch
   ```

3. **Run Diagnostic Task**
   ```bash
   # Using uv (no need to activate venv)
   uv run python diagnostic_task.py
   
   # Or activate venv first
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Mac/Linux
   python diagnostic_task.py
   ```

4. **Complete Levels**
   - **Level 1**: Get basic text generation working (required)
   - **Level 2**: Experiment with parameters and document findings
   - **Level 3**: Pick one challenge (A, B, C, or D), feel free to do all of them!!
   - **Level 4**: Build something new and creative

## Task Details

See individual level folders for detailed instructions:
- [Level 1](level1/README.md) - Install & Run
- [Level 2](level2/README.md) - Experiment & Document
- [Level 3](level3/README.md) - Open-Ended Challenge
- [Level 4](level4/README.md) - Freestyle Build

## Submission

See [submission_format.txt](submission_format.txt) for the submission template.

**Due:** Wednesday 11:59 PM  
**Submit to:** Teams #architecture channel

## Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Pipeline Guide](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [Hugging Face Models](https://huggingface.co/models)
- [Diagnostic Repository](https://github.com/Marzooqad/diagnostic-SotonLM-)

## Notes

- This project uses **uv** for fast dependency management
- The `.venv/` folder is excluded from git (see `.gitignore`)
- Generated output files (like `results.txt`) should be created locally
- See `SETUP.MD` for detailed setup instructions
- See `GITHUB_SETUP.md` for instructions on pushing to GitHub
- Dependencies are managed in `pyproject.toml` (not requirements.txt)

## Repository Structure

This folder is part of the [diagnostic repository](https://github.com/Marzooqad/diagnostic-SotonLM-). 
Follow the guidelines in `GITHUB_SETUP.md` to push your work.

