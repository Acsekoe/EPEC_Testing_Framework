# EPEC_Testing_Framework

## Setup (Windows)

Create the venv outside the repo (matches `.vscode/settings.json`):

```powershell
py -3 -m venv "$env:USERPROFILE\.venvs\EEG_EPEC"
& "$env:USERPROFILE\.venvs\EEG_EPEC\Scripts\python.exe" -m pip install --upgrade pip
& "$env:USERPROFILE\.venvs\EEG_EPEC\Scripts\pip.exe" install -r requirements.txt
```

Open a new VS Code terminal and the `EEG_EPEC` venv should auto-activate.

Notes:
- Gurobi requires a valid license. If `gurobipy` installs but the solver is still unavailable, run `gurobi_cl --version` to confirm the CLI is on PATH and verify your license setup.
