from typer.testing import CliRunner

from qopd import __version__
from qopd.cli import app

runner = CliRunner()


def test_help():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in ("env", "status", "peer"):
        assert cmd in res.output


def test_version():
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0
    assert __version__ in res.output


def test_env():
    res = runner.invoke(app, ["env"])
    assert res.exit_code == 0, res.output
    assert "Environment report" in res.output
    assert "unified memory" in res.output


def test_status_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("QOPD_RUNS", str(tmp_path))
    res = runner.invoke(app, ["status"])
    assert res.exit_code == 0
    assert "no runs" in res.output


def test_config_show_override():
    res = runner.invoke(app, ["config", "show", "configs/train/opd.yaml", "rounds=5"])
    assert res.exit_code == 0, res.output
    assert "rounds: 5" in res.output


def test_config_check():
    res = runner.invoke(app, ["config", "check"])
    assert res.exit_code == 0, res.output
    assert "FAIL" not in res.output
