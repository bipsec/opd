"""Allow `python -m qopd ...` (the container's `qopd` wrapper calls this)."""

from qopd.cli import app

app(prog_name="qopd")
