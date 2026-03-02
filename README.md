Quick setup

1. Install Python 3.10+ from https://www.python.org/downloads/ and ensure `python` is in PATH.

2. Create a virtual environment and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# or
.\.venv\Scripts\activate.bat   # cmd
```

3. Install dependencies:

```powershell
python -m pip install -U pip
pip install -r requirements.txt
```

Notes:
- `torch` installation may require selecting the correct build for your OS/GPU. See https://pytorch.org/get-started/locally/ for specific install commands.
- If you don't plan to train with `torchrl`/`tensordict`, you can stub out or adapt code.

Run:

```powershell
python "SIM\SIMTEST.py"
```
