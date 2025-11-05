# CIS6020 — 5‑Minute Setup

Follow these steps. Copy/paste the commands.

## Quick Start

1) Install UV (macOS/Linux): `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2) Get the code and go to the folder:
   - `git clone https://github.com/klukaszek/MRI-RayTracer`
   - `cd MRI-RayTracer`
3) Install deps (pick one):
   - CPU (most laptops): `uv sync --extra cpu`
   - CUDA 12.8 (NVIDIA/remote GPU): `uv sync --extra cu128`
   - Note: `uv sync` automatically installs Python 3.11.11 if needed (no extra steps).
4) Open JupyterLab: `uv run jupyter lab` → open `notebooks/sample_notebook/interactive.ipynb` and Run All.

Done. You don’t need to “activate” anything if you use `uv run ...`.

## VS Code (optional)

- Install extensions: Python, Jupyter
- Command Palette → “Python: Select Interpreter” → pick `.venv`
- Open a `.ipynb` and run cells

## CLI Examples

- Torch/Vision check: `uv run python -c "import torch, torchvision; print(torch.__version__, 'vision OK')"`
- CUDA check (GPU host): `uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`

## Troubleshooting (fast)

- UV not found: reopen terminal after install; `uv --version`
- Kernel not found: start with `uv run jupyter lab` or select `.venv` in VS Code
- Wrong Python: `uv python install 3.11.11 && uv sync --reinstall`; check `uv run python -V`
