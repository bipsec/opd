"""Allow `python -m qopd ...` (the container's `qopd` wrapper calls this)."""

from qopd.cli import app

if __name__ == "__main__":  # guard: vLLM worker processes may re-import this module
    app(prog_name="qopd")
