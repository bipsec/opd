import csv
import time

from qopd.utils.monitor import COLUMNS, Monitor


def test_monitor_writes_rows(tmp_path):
    path = tmp_path / "monitor.csv"
    with Monitor(path, interval_s=0.05, power_fn=lambda: 42.0) as mon:
        time.sleep(0.3)
    rows = list(csv.reader(path.open()))
    assert tuple(rows[0]) == COLUMNS
    assert len(rows) - 1 == mon.samples >= 2
    first = dict(zip(COLUMNS, rows[1], strict=True))
    assert float(first["power_w"]) == 42.0
    assert float(first["mem_total_gb"]) > 0
    assert float(first["proc_rss_gb"]) > 0


def test_monitor_appends_without_second_header(tmp_path):
    path = tmp_path / "monitor.csv"
    for _ in range(2):
        with Monitor(path, interval_s=0.05, power_fn=lambda: 1.0):
            time.sleep(0.1)
    lines = path.read_text().splitlines()
    assert sum(1 for line in lines if line.startswith("t_unix")) == 1
