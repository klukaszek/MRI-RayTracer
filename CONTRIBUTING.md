# Contributing — Simple Branch Workflow

Follow these steps exactly to work safely on your own branch. Do not push to `main` directly.

## Quick Workflow (copy/paste)

Before you start, install UV (one-time):
- macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

1) Clone the repo and enter the folder
- `git clone https://github.com/klukaszek/MRI-RayTracer`
- `cd MRI-RayTracer`

2) Set up the environment once (see README for details)
- CPU (most laptops): `uv sync --extra cpu`
- CUDA 12.8 (GPU/remote): `uv sync --extra cu128`

3) Create your own branch from `main`
- Make sure you’re on main and up to date:
  - `git switch main`
  - `git pull`
- Create a new branch (pick one):
  - Feature: `git switch -c feature/<short-name>`
  - Fix: `git switch -c fix/<short-name>`
  - Notebook: `git switch -c nb/<short-name>`

4) Make changes locally
- Notebooks: prefer small outputs. In Jupyter, “Kernel → Restart & Clear All Outputs” before saving if big images are produced.
- Don’t commit data, secrets, or large binaries.

5) Save and commit
- Stage everything: `git add -A`
- Commit: `git commit -m "<clear message>"`

6) Push your branch to GitHub
- `git push -u origin <your-branch>`

7) Open a Pull Request (PR)
- Go to the GitHub repo → “Compare & pull request” → target branch is `main`.
- Keep the PR small, describe what changed, and how to run it.

8) Keep your branch up to date (if main changes)
- `git switch main && git pull`
- `git switch <your-branch> && git merge main`
- Resolve conflicts if prompted, then:
  - `git add -A && git commit` (to finish the merge)

9) After review
- Make updates on your branch → push again: `git push`
- When approved, squash/merge via the GitHub button (or a maintainer will).

## Tips
- VS Code has a friendly Git UI (Source Control icon). It shows changes, lets you commit, and helps resolve conflicts.
- Use `uv run ...` to execute Python without manually activating the venv, e.g.:
  - `uv run jupyter lab`
  - `uv run python scripts/validation/validate_datasets.py`
- If you get stuck, share your branch name and a short description of what you tried.

## Do / Don’t
- Do: one branch per task; clear commit messages; small PRs.
- Do: run notebooks/scripts to verify before opening a PR.
- Don’t: push to `main`; commit large data; commit secrets.
