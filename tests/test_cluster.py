import subprocess
from pathlib import Path

from qopd.utils.cluster import _askpass_script, load_nodes, parse_status


def test_load_nodes():
    nodes, env_file = load_nodes()
    assert {"local", "peer"} <= set(nodes)
    assert nodes["local"].role == "primary"
    assert env_file.name == ".env"


def test_askpass_reads_but_does_not_contain_password(tmp_path):
    env = tmp_path / ".env"
    env.write_text('PASSWD="s3cr3t"\n')
    script = _askpass_script(env, tmp_path)
    assert "s3cr3t" not in script.read_text()
    out = subprocess.run([str(script)], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "s3cr3t"


def test_parse_status():
    out = "NVIDIA GB10\n---\nMemAvailable:   59256236 kB\n---\n629616, python, 56243 MiB\n"
    st = parse_status("peer", out)
    assert st.gpu == "NVIDIA GB10"
    assert st.mem_available_gb == 56.5
    assert st.gpu_processes == ["629616, python, 56243 MiB"]
    assert not st.free
    assert parse_status("peer", "NVIDIA GB10\n---\nMemAvailable: 1048576 kB\n---\n").free


def test_cluster_yaml_exists():
    assert (Path(__file__).resolve().parents[1] / "configs" / "cluster.yaml").exists()
