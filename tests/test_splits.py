"""PLAN 2.4: splits are disjoint, deterministic, and the manifest hashes match the files."""

import pytest

from qopd.config import DataCfg
from qopd.data.sources import read_parquet, write_parquet
from qopd.data.splits import (
    ManifestMismatchError,
    Splits,
    check_disjoint,
    ids_sha256,
    make_splits,
    verify_manifest,
    write_manifest,
)
from qopd.types import Problem

POOL = [Problem(f"src/{i:05d}", f"problem {i}", str(i), "src") for i in range(300)]
SIZES = {"n_train": 200, "n_coldstart": 50, "n_calibration": 30, "n_probe": 20}


def test_sizes_subset_and_disjoint():
    s = make_splits(POOL, seed=0, **SIZES)
    assert (len(s.train), len(s.coldstart), len(s.calibration), len(s.probe)) == (200, 50, 30, 20)
    assert s.n_reserve == 50
    tr = {p.id for p in s.train}
    assert {p.id for p in s.coldstart} <= tr
    assert not tr & {p.id for p in s.calibration}
    assert not tr & {p.id for p in s.probe}
    assert not {p.id for p in s.calibration} & {p.id for p in s.probe}


def test_deterministic_and_order_independent():
    a = make_splits(POOL, seed=0, **SIZES)
    b = make_splits(list(reversed(POOL)), seed=0, **SIZES)
    c = make_splits(POOL, seed=1, **SIZES)
    assert [p.id for p in a.train] == [p.id for p in b.train]
    assert [p.id for p in a.train] != [p.id for p in c.train]


def test_stable_under_removal():
    a = make_splits(POOL, seed=0, **SIZES)
    removed = {a.train[3].id, a.probe[0].id}
    b = make_splits([p for p in POOL if p.id not in removed], seed=0, **SIZES)
    a_train = [p.id for p in a.train if p.id not in removed]
    assert [p.id for p in b.train][: len(a_train)] == a_train  # survivors keep their place
    assert {p.id for p in a.coldstart} - removed <= {p.id for p in b.coldstart} | {a.train[50].id}


def test_check_disjoint_catches_overlap():
    s = make_splits(POOL, seed=0, **SIZES)
    with pytest.raises(ValueError, match="share"):
        check_disjoint(Splits(s.train, s.coldstart, s.calibration + s.train[:1], s.probe))
    with pytest.raises(ValueError, match="subset"):
        check_disjoint(Splits(s.train, s.coldstart + s.probe[:1], s.calibration, s.probe))


def test_pool_too_small():
    with pytest.raises(ValueError, match="pool has"):
        make_splits(POOL[:100], seed=0, **SIZES)


def test_manifest_round_trip_and_tamper(tmp_path):
    pool_file = tmp_path / "pool.parquet"
    write_parquet(POOL, pool_file)
    ev = tmp_path / "math500.parquet"
    write_parquet(POOL[:5], ev)
    s = make_splits(POOL, seed=0, **SIZES)
    man = tmp_path / "manifest.json"
    m = write_manifest(s, tmp_path / "splits", man, DataCfg(), pool_file, {"math500": ev})
    assert m["splits"]["train"]["n"] == 200
    assert m["splits"]["probe"]["ids_sha256"] == ids_sha256(s.probe)
    assert read_parquet(tmp_path / "splits" / "coldstart.parquet") == s.coldstart
    verify_manifest(man, root=tmp_path)  # hashes match
    write_parquet(s.probe[:-1], tmp_path / "splits" / "probe.parquet")  # tamper
    with pytest.raises(ManifestMismatchError, match="probe"):
        verify_manifest(man, root=tmp_path)
