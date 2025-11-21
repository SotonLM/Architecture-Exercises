# GitHub Setup Guide

This guide will help you push your work to the diagnostic repository.

## Option 1: Clone and Copy (Recommended)

1. **Clone the repository:**
   ```bash
   cd ..
   git clone https://github.com/Marzooqad/diagnostic-SotonLM-.git
   cd diagnostic-SotonLM-
   ```

2. **Create your branch:**
   ```bash
   git checkout -b SotonLM-diagnostic-architecture
   ```

3. **Copy your week3 folder:**
   ```bash
   # Copy the entire week3 folder into the repository
   cp -r ../week3 .  # Linux/Mac
   # Or use Windows Explorer to copy the folder
   ```

4. **Add and commit:**
   ```bash
   git add week3/
   git commit -m "Add week3 diagnostic task"
   ```

5. **Push to GitHub:**
   ```bash
   git push origin SotonLM-diagnostic-architecture
   ```

6. **Create a Pull Request:**
   - Go to https://github.com/Marzooqad/diagnostic-SotonLM-
   - Click "New Pull Request"
   - Select your branch
   - Add description and submit

## Option 2: Initialize Git Here

If you want to initialize git in this folder:

1. **Initialize git:**
   ```bash
   git init
   ```

2. **Add remote:**
   ```bash
   git remote add origin https://github.com/Marzooqad/diagnostic-SotonLM-.git
   ```

3. **Create branch:**
   ```bash
   git checkout -b SotonLM-diagnostic-architecture
   ```

4. **Add files:**
   ```bash
   git add .
   git commit -m "Add week3 diagnostic task"
   ```

5. **Push:**
   ```bash
   git push -u origin SotonLM-diagnostic-architecture
   ```

## Files Included

The following files are ready to be committed:
- ✅ `README.md` - Project documentation
- ✅ `.gitignore` - Excludes venv, cache files, etc.
- ✅ `pyproject.toml` - Python dependencies (for uv)
- ✅ `diagnostic_task.py` - Main task file
- ✅ `SETUP.MD` - Setup instructions
- ✅ `task.txt` - Complete task description
- ✅ `submission_format.txt` - Submission template
- ✅ `level1/`, `level2/`, `level3/`, `level4/` - Task folders

## Files Excluded (via .gitignore)

- `venv/` - Virtual environment
- `__pycache__/` - Python cache
- `*.onetoc2` - OneNote files
- `*.log` - Log files
- Generated output files

## Best Practices

1. **Commit Message Format:**
   ```
   Add week3 diagnostic task
   
   - Includes all 4 levels of tasks
   - Organized in separate folders
   - Uses uv for dependency management
   - Includes setup instructions and pyproject.toml
   ```

2. **Branch Naming:**
   - Branch name: `SotonLM-diagnostic-architecture`
   - Use descriptive names for other branches
   - Avoid generic names like `update` or `fix`

3. **Before Pushing:**
   - Make sure all files are properly organized
   - Test that `diagnostic_task.py` runs
   - Verify README.md is accurate

## Troubleshooting

- **Permission denied / Authentication errors:** 
  - Make sure you've forked the repository on GitHub first
  - Ensure you're pushing to your fork, not the original repo
  - Check that you're authenticated with GitHub (may need to set up SSH keys or use a personal access token)
- **Branch already exists:** Use a different branch name or delete the old one
- **Merge conflicts:** Pull latest changes first: `git pull origin main`

