"""F23 public Python contract for binary ``ClusteringResult`` checkpoints."""

from __future__ import annotations

import hashlib
import struct

import pytest

import dtwcpp
import dtwcpp._dtwcpp_core as core
from dtwcpp import load_binary_checkpoint, save_binary_checkpoint


LABELS = (2, 1, 0, 2, 2, 0, 0)
MEDOIDS = (6, 1, 4)
TOTAL_COST = -13.25
ITERATIONS = 0x01020304
WIRE_FORMAT = "<4sHHiiiB3xd3i7i"
WIRE_HEX = (
    "44434b5001000000030000000700000004030201010000000000000000802ac0"
    "0600000001000000040000000200000001000000000000000200000002000000"
    "0000000000000000"
)
WIRE_SHA256 = "DC832EDBD214FD847B7EC8BC57884881F1CEA196139FAC7DD5B939D5E7CD1A98"


def _independent_wire_oracle() -> bytes:
    return struct.pack(
        WIRE_FORMAT,
        b"DCKP",
        1,
        0,
        len(MEDOIDS),
        len(LABELS),
        ITERATIONS,
        1,
        TOTAL_COST,
        *MEDOIDS,
        *LABELS,
    )


def _result_fixture() -> dtwcpp.ClusteringResult:
    result = dtwcpp.ClusteringResult()
    result.labels = list(LABELS)
    result.medoid_indices = list(MEDOIDS)
    result.total_cost = TOTAL_COST
    result.iterations = ITERATIONS
    result.converged = True
    return result


def _matching_field_count(result: dtwcpp.ClusteringResult) -> int:
    return sum(
        (
            tuple(result.labels) == LABELS,
            tuple(result.medoid_indices) == MEDOIDS,
            result.total_cost == TOTAL_COST,
            result.iterations == ITERATIONS,
            result.converged is True,
        )
    )


def _assert_read_io_error(path: object) -> int:
    with pytest.raises(dtwcpp.IOError) as caught:
        load_binary_checkpoint(path)
    assert type(caught.value) is dtwcpp.IOError
    assert isinstance(caught.value, dtwcpp.DtwcError)
    assert isinstance(caught.value, OSError)
    assert str(caught.value) == (
        "load_binary_checkpoint: cannot read a valid binary result "
        f"checkpoint from '{path}'."
    )
    return 1


def test_binary_checkpoint_exact_wire_roundtrip_and_io_errors(tmp_path):
    """Drive both bound C++ readers against an independent non-degenerate wire."""
    checkpoint = tmp_path / "result.dckp"
    oracle = _independent_wire_oracle()
    assert len(oracle) == 72
    assert oracle.hex() == WIRE_HEX
    assert hashlib.sha256(oracle).hexdigest().upper() == WIRE_SHA256

    writer_return = save_binary_checkpoint(_result_fixture(), checkpoint)
    assert writer_return is None
    emitted = checkpoint.read_bytes()
    assert emitted == oracle
    assert len(emitted) == 72
    assert hashlib.sha256(emitted).hexdigest().upper() == WIRE_SHA256
    byte_matches = sum(actual == expected for actual, expected in zip(emitted, oracle))
    assert byte_matches == 72

    cpp_reader_calls = 0
    loaded_public = load_binary_checkpoint(str(checkpoint))
    assert type(loaded_public) is dtwcpp.ClusteringResult
    cpp_reader_calls += 1
    loaded_core = core.load_binary_checkpoint(checkpoint)
    assert type(loaded_core) is dtwcpp.ClusteringResult
    cpp_reader_calls += 1
    field_matches = _matching_field_count(loaded_public)
    field_matches += _matching_field_count(loaded_core)
    assert field_matches == 10
    assert cpp_reader_calls == 2

    missing = tmp_path / "missing.dckp"
    malformed = tmp_path / "malformed.dckp"
    malformed.write_bytes(oracle[:-1])
    io_errors = _assert_read_io_error(missing)
    io_errors += _assert_read_io_error(str(malformed))

    blocked_parent = tmp_path / "blocked-parent"
    blocked_child = blocked_parent / "result.dckp"
    blocked_bytes = b"not a directory"
    blocked_parent.write_bytes(blocked_bytes)
    with pytest.raises(dtwcpp.IOError) as caught:
        save_binary_checkpoint(_result_fixture(), blocked_child)
    assert type(caught.value) is dtwcpp.IOError
    assert isinstance(caught.value, dtwcpp.DtwcError)
    assert isinstance(caught.value, OSError)
    assert blocked_parent.read_bytes() == blocked_bytes
    assert not blocked_child.exists()
    io_errors += 1
    assert io_errors == 3

    export_matches = sum(
        (
            dtwcpp.save_binary_checkpoint is core.save_binary_checkpoint,
            dtwcpp.load_binary_checkpoint is core.load_binary_checkpoint,
        )
    )
    assert export_matches == 2
    print(
        "F23_PYTHON_CHECKPOINT "
        f"exports={export_matches}/2 fields={field_matches}/10 "
        f"bytes={byte_matches}/72 cpp_reader={cpp_reader_calls}/2 "
        f"io_errors={io_errors}/3 skips=0 verdict=PASS"
    )


def test_binary_checkpoint_public_exports_are_core_identities():
    assert save_binary_checkpoint is core.save_binary_checkpoint
    assert load_binary_checkpoint is core.load_binary_checkpoint
    assert dtwcpp.save_binary_checkpoint is core.save_binary_checkpoint
    assert dtwcpp.load_binary_checkpoint is core.load_binary_checkpoint


def test_binary_checkpoint_all_exports_are_unique():
    assert dtwcpp.__all__.count("save_binary_checkpoint") == 1
    assert dtwcpp.__all__.count("load_binary_checkpoint") == 1
