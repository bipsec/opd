import pytest

from qopd.utils.run import (
    CONTRACT_DIRS,
    CONTRACT_FILES,
    RunConfigMismatchError,
    create_run_dir,
)


def test_contract_files(tmp_path):
    rd = create_run_dir("exp", "r0", {"a": 1, "b": {"c": 2}}, root=tmp_path)
    assert rd.path == tmp_path / "exp" / "r0"
    for name in CONTRACT_FILES:
        assert (rd.path / name).is_file(), name
    for name in CONTRACT_DIRS:
        assert (rd.path / name).is_dir(), name
    assert rd.missing() == []
    assert rd.load_config() == {"a": 1, "b": {"c": 2}}
    assert (rd.path / "git_sha.txt").read_text().strip()
    assert rd.monitor.read_text().startswith("t_unix,")
    assert rd.load_state()["status"] == "created"


def test_done_only_after_mark(tmp_path):
    rd = create_run_dir("exp", "r0", {"a": 1}, root=tmp_path)
    assert not rd.is_done
    rd.mark_done()
    assert rd.is_done
    assert rd.load_state()["status"] == "done"


def test_resume_and_mismatch(tmp_path):
    rd = create_run_dir("exp", "r0", {"a": 1}, root=tmp_path)
    rd.log_metrics({"round": 0, "reward": 0.5})
    again = create_run_dir("exp", "r0", {"a": 1}, root=tmp_path)
    assert again.load_state()["resumed"] is True
    assert again.metrics.read_text().count("\n") == 1  # metrics kept
    with pytest.raises(RunConfigMismatchError):
        create_run_dir("exp", "r0", {"a": 2}, root=tmp_path)
