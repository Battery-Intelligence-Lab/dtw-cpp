"""F22: frozen Python compatibility aliases warn once and preserve behaviour.

This file deliberately drives the public ``dtwcpp`` surface.  Legacy spellings
belong here (and in the binding/contract history), not in ordinary examples or
tests.  The registered inventory is 12 symbols and 13 primary operations
because ``Problem.n_repetition`` has distinct read and write operations.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable
import warnings

import numpy as np
import pytest

import dtwcpp


ROOT = Path(__file__).resolve().parents[2]
THIS_FILE = Path(__file__).resolve()

_SERIES = [[0.0], [1.0], [3.0], [10.0], [12.0], [15.0]]
_NAMES = [f"s{i}" for i in range(len(_SERIES))]
_DISTANCE_MATRIX = np.abs(
    np.subtract.outer(
        np.asarray(_SERIES, dtype=np.float64)[:, 0],
        np.asarray(_SERIES, dtype=np.float64)[:, 0],
    )
)
_LABELS_TRUE = [0, 0, 0, 1, 1, 1, 2, 2]
_LABELS_PRED = [0, 0, 1, 1, 1, 2, 2, 2]

_PRIMARY_PASSED: set[str] = set()
_CANONICAL_SILENT: set[str] = set()
_EQUIVALENT: set[str] = set()
_CLASS_ROUTES: set[str] = set()
_IDENTITY: set[str] = set()
_IMPORT_SILENT = False
_ORDINARY_HYGIENE = False


SubjectFactory = Callable[[], tuple[Any, Any]]
Operation = Callable[[Any], Any]
Comparator = Callable[[Any, Any, Any, Any], None]
Snapshot = Callable[[Any], Any]


@dataclass(frozen=True)
class AliasCase:
    case_id: str
    old: str
    new: str
    make_subjects: SubjectFactory
    canonical: Operation
    legacy: Operation
    equivalent: Comparator
    snapshot: Snapshot | None = None

    @property
    def message(self) -> str:
        return f"{self.old} is deprecated; use {self.new}"


def _problem(name: str) -> Any:
    problem = dtwcpp.Problem(name)
    problem.set_data(_SERIES, _NAMES)
    return problem


def _problem_pair() -> tuple[Any, Any]:
    return _problem("f22-canonical"), _problem("f22-legacy")


def _repetition_pair() -> tuple[Any, Any]:
    canonical, legacy = dtwcpp.Problem("f22-canonical"), dtwcpp.Problem("f22-legacy")
    canonical.n_repetitions = 7
    legacy.n_repetitions = 7
    return canonical, legacy


def _cluster_count_pair() -> tuple[Any, Any]:
    canonical, legacy = dtwcpp.Problem("f22-canonical"), dtwcpp.Problem("f22-legacy")
    canonical.set_n_clusters(3)
    legacy.set_n_clusters(3)
    return canonical, legacy


def _matrix_pair() -> tuple[Any, Any]:
    return _problem_pair()


def _scoring_problem(name: str) -> Any:
    problem = _problem(name)
    problem.set_distance_matrix(_DISTANCE_MATRIX)
    problem.set_n_clusters(2)
    problem.clusters_ind = [0, 0, 0, 1, 1, 1]
    problem.centroids_ind = [1, 4]
    return problem


def _scoring_pair() -> tuple[Any, Any]:
    return _scoring_problem("f22-canonical"), _scoring_problem("f22-legacy")


def _label_pair() -> tuple[Any, Any]:
    labels = (_LABELS_TRUE, _LABELS_PRED)
    return labels, labels


def _result_pair() -> tuple[Any, Any]:
    kwargs = {
        "device": "cpu",
        "elapsed_s": 0.0,
        "k": 2,
        "n_series": 4,
        "medoid_indices": [0, 2],
    }
    return (
        dtwcpp.Result([0, 0, 1, 1], **kwargs),
        dtwcpp.Result([0, 0, 1, 1], **kwargs),
    )


def _result_constructor_pair() -> tuple[Any, Any]:
    args = ([0, 0, 1, 1],)
    kwargs = {
        "device": "cpu",
        "elapsed_s": 0.0,
        "k": 2,
        "n_series": 4,
        "medoid_indices": [0, 2],
    }
    subject = (args, kwargs)
    return subject, subject


def _canonical_set_clusters(problem: Any) -> tuple[Any, int]:
    result = problem.set_n_clusters(3)
    return result, problem.n_clusters()


def _legacy_set_clusters(problem: Any) -> tuple[Any, int]:
    result = problem.set_number_of_clusters(3)
    return result, problem.n_clusters()


def _canonical_set_repetition(problem: Any) -> tuple[Any, int]:
    result = setattr(problem, "n_repetitions", 11)
    return result, problem.n_repetitions


def _legacy_set_repetition(problem: Any) -> tuple[Any, int]:
    result = setattr(problem, "n_repetition", 11)
    return result, problem.n_repetitions


def _canonical_write_matrix(problem: Any) -> tuple[Any, np.ndarray]:
    result = problem.set_distance_matrix(_DISTANCE_MATRIX)
    return result, problem.distance_matrix()


def _legacy_write_matrix(problem: Any) -> tuple[Any, np.ndarray]:
    result = problem.set_distance_matrix_from_numpy(_DISTANCE_MATRIX)
    return result, problem.distance_matrix()


def _canonical_result(subject: Any) -> Any:
    args, kwargs = subject
    return dtwcpp.Result(*args, **kwargs)


def _legacy_result(subject: Any) -> tuple[type, Any]:
    args, kwargs = subject
    result_type = dtwcpp.ClusterResult
    return result_type, result_type(*args, **kwargs)


def _equal(
    canonical_subject: Any,
    legacy_subject: Any,
    canonical: Any,
    legacy: Any,
) -> None:
    del canonical_subject, legacy_subject
    assert canonical == legacy


def _equal_matrix(
    canonical_subject: Any,
    legacy_subject: Any,
    canonical: np.ndarray,
    legacy: np.ndarray,
) -> None:
    del canonical_subject
    np.testing.assert_array_equal(canonical, _DISTANCE_MATRIX)
    np.testing.assert_array_equal(legacy, canonical)
    independent = legacy_subject.distance_matrix()
    np.testing.assert_array_equal(independent, legacy)
    assert independent is not legacy
    assert not np.shares_memory(independent, legacy)


def _equal_written_matrix(
    canonical_subject: Any,
    legacy_subject: Any,
    canonical: tuple[Any, np.ndarray],
    legacy: tuple[Any, np.ndarray],
) -> None:
    del canonical_subject, legacy_subject
    assert canonical[0] is None
    assert legacy[0] is None
    np.testing.assert_array_equal(legacy[1], canonical[1])


def _equal_result_constructor(
    canonical_subject: Any,
    legacy_subject: Any,
    canonical: Any,
    legacy: tuple[type, Any],
) -> None:
    del canonical_subject, legacy_subject
    result_type, result = legacy
    assert result_type is dtwcpp.Result
    assert type(result) is dtwcpp.Result
    assert type(canonical) is dtwcpp.Result
    np.testing.assert_array_equal(result.labels, canonical.labels)
    np.testing.assert_array_equal(result.medoids, canonical.medoids)
    assert result.device == canonical.device
    assert result.k == canonical.k
    assert result.n_series == canonical.n_series


def _equal_result_medoids(
    canonical_subject: Any,
    legacy_subject: Any,
    canonical: np.ndarray,
    legacy: np.ndarray,
) -> None:
    del canonical_subject
    np.testing.assert_array_equal(legacy, canonical)
    assert legacy is legacy_subject.medoids


def _set_k_snapshot(problem: Any) -> int:
    return problem.n_clusters()


def _repetition_snapshot(problem: Any) -> int:
    return problem.n_repetitions


def _matrix_snapshot(problem: Any) -> bool:
    return problem.is_distance_matrix_filled()


_CASES = [
    AliasCase(
        "set_number_of_clusters",
        "Problem.set_number_of_clusters",
        "Problem.set_n_clusters",
        _problem_pair,
        _canonical_set_clusters,
        _legacy_set_clusters,
        _equal,
        _set_k_snapshot,
    ),
    AliasCase(
        "n_repetition_get",
        "Problem.n_repetition",
        "Problem.n_repetitions",
        _repetition_pair,
        lambda problem: problem.n_repetitions,
        lambda problem: problem.n_repetition,
        _equal,
    ),
    AliasCase(
        "n_repetition_set",
        "Problem.n_repetition",
        "Problem.n_repetitions",
        _repetition_pair,
        _canonical_set_repetition,
        _legacy_set_repetition,
        _equal,
        _repetition_snapshot,
    ),
    AliasCase(
        "cluster_size",
        "Problem.cluster_size",
        "Problem.n_clusters",
        _cluster_count_pair,
        lambda problem: problem.n_clusters(),
        lambda problem: problem.cluster_size,
        _equal,
    ),
    AliasCase(
        "distance_matrix_numpy",
        "Problem.distance_matrix_numpy",
        "Problem.distance_matrix",
        _matrix_pair,
        lambda problem: problem.distance_matrix(),
        lambda problem: problem.distance_matrix_numpy(),
        _equal_matrix,
        _matrix_snapshot,
    ),
    AliasCase(
        "set_distance_matrix_from_numpy",
        "Problem.set_distance_matrix_from_numpy",
        "Problem.set_distance_matrix",
        _problem_pair,
        _canonical_write_matrix,
        _legacy_write_matrix,
        _equal_written_matrix,
        _matrix_snapshot,
    ),
    AliasCase(
        "davies_bouldin_index",
        "dtwcpp.davies_bouldin_index",
        "dtwcpp.davies_bouldin",
        _scoring_pair,
        dtwcpp.davies_bouldin,
        dtwcpp.davies_bouldin_index,
        _equal,
    ),
    AliasCase(
        "dunn_index",
        "dtwcpp.dunn_index",
        "dtwcpp.dunn",
        _scoring_pair,
        dtwcpp.dunn,
        dtwcpp.dunn_index,
        _equal,
    ),
    AliasCase(
        "calinski_harabasz_index",
        "dtwcpp.calinski_harabasz_index",
        "dtwcpp.calinski_harabasz",
        _scoring_pair,
        dtwcpp.calinski_harabasz,
        dtwcpp.calinski_harabasz_index,
        _equal,
    ),
    AliasCase(
        "adjusted_rand_index",
        "dtwcpp.adjusted_rand_index",
        "dtwcpp.adjusted_rand",
        _label_pair,
        lambda labels: dtwcpp.adjusted_rand(*labels),
        lambda labels: dtwcpp.adjusted_rand_index(*labels),
        _equal,
    ),
    AliasCase(
        "normalized_mutual_information",
        "dtwcpp.normalized_mutual_information",
        "dtwcpp.normalized_mutual_info",
        _label_pair,
        lambda labels: dtwcpp.normalized_mutual_info(*labels),
        lambda labels: dtwcpp.normalized_mutual_information(*labels),
        _equal,
    ),
    AliasCase(
        "ClusterResult",
        "dtwcpp.ClusterResult",
        "dtwcpp.Result",
        _result_constructor_pair,
        _canonical_result,
        _legacy_result,
        _equal_result_constructor,
    ),
    AliasCase(
        "Result.medoid_indices",
        "Result.medoid_indices",
        "Result.medoids",
        _result_pair,
        lambda result: result.medoids,
        lambda result: result.medoid_indices,
        _equal_result_medoids,
    ),
]


@pytest.fixture(scope="session", autouse=True)
def _print_green_marker():
    yield
    complete = (
        _PRIMARY_PASSED == {case.case_id for case in _CASES}
        and _CANONICAL_SILENT == _PRIMARY_PASSED
        and _EQUIVALENT == _PRIMARY_PASSED
        and _CLASS_ROUTES == {"lookup", "from_import", "star_import"}
        and _IDENTITY == {"ClusterResult", "Result.medoid_indices"}
        and _IMPORT_SILENT
        and _ORDINARY_HYGIENE
    )
    if complete:
        print(
            "F22_PYTHON_GATE alias_symbols=12 operations=13 "
            "primary_warn_once=13 canonical_silent=13 equivalent=13 "
            "class_routes=3 identity=2 ordinary_legacy=0 verdict=PASS"
        )


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.case_id)
def test_retained_alias_warns_once_is_caller_attributed_and_equivalent(case):
    """Drive every public compatibility operation twice from a user frame."""
    exact_message = f"^{re.escape(case.message)}$"

    for _ in range(2):
        canonical_subject, legacy_subject = case.make_subjects()

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            canonical = case.canonical(canonical_subject)

        with pytest.warns(DeprecationWarning, match=exact_message) as caught:
            legacy = case.legacy(legacy_subject)
            case.equivalent(
                canonical_subject, legacy_subject, canonical, legacy
            )

        assert len(caught) == 1
        assert caught[0].category is DeprecationWarning
        assert str(caught[0].message) == case.message
        assert Path(caught[0].filename).resolve() == THIS_FILE

    if case.snapshot is not None:
        _, legacy_subject = case.make_subjects()
        before = case.snapshot(legacy_subject)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with pytest.raises(DeprecationWarning, match=exact_message):
                case.legacy(legacy_subject)
        assert case.snapshot(legacy_subject) == before

    _PRIMARY_PASSED.add(case.case_id)
    _CANONICAL_SILENT.add(case.case_id)
    _EQUIVALENT.add(case.case_id)
    if case.case_id in {"ClusterResult", "Result.medoid_indices"}:
        _IDENTITY.add(case.case_id)


def _resolve_cluster_result(route: str) -> type:
    if route == "lookup":
        return getattr(dtwcpp, "ClusterResult")

    namespace: dict[str, Any] = {}
    if route == "from_import":
        source = "from dtwcpp import ClusterResult\nresolved = ClusterResult\n"
    else:
        source = "from dtwcpp import *\nresolved = ClusterResult\n"
    exec(compile(source, str(THIS_FILE), "exec"), namespace)
    return namespace["resolved"]


@pytest.mark.parametrize("route", ["lookup", "from_import", "star_import"])
def test_cluster_result_supplemental_resolution_routes(route):
    """Every public name-resolution route warns and returns the exact class."""
    message = "dtwcpp.ClusterResult is deprecated; use dtwcpp.Result"

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        canonical = dtwcpp.Result

    for _ in range(2):
        with pytest.warns(
            DeprecationWarning, match=f"^{re.escape(message)}$"
        ) as caught:
            resolved = _resolve_cluster_result(route)
            assert resolved is canonical
        assert len(caught) == 1
        assert caught[0].category is DeprecationWarning
        assert str(caught[0].message) == message
        assert Path(caught[0].filename).resolve() == THIS_FILE

    _CLASS_ROUTES.add(route)


def test_plain_package_import_is_deprecation_silent():
    """A canonical package import must not resolve any deprecated name."""
    global _IMPORT_SILENT

    script = (
        "import warnings\n"
        "warnings.simplefilter('error', DeprecationWarning)\n"
        "import dtwcpp\n"
        "assert dtwcpp.Result.__name__ == 'Result'\n"
    )
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "2")
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "DeprecationWarning" not in completed.stderr
    _IMPORT_SILENT = True


_LEGACY_CALL_NAMES = {
    "set_number_of_clusters",
    "distance_matrix_numpy",
    "set_distance_matrix_from_numpy",
    "davies_bouldin_index",
    "dunn_index",
    "calinski_harabasz_index",
    "adjusted_rand_index",
    "normalized_mutual_information",
    "ClusterResult",
}

_LEGACY_ATTRIBUTE_NAMES = {
    "n_repetition",
    "cluster_size",
}


def _ordinary_legacy_calls() -> list[str]:
    violations: set[str] = set()
    scan_roots = [
        ROOT / "python",
        ROOT / "tests",
        ROOT / "examples",
        ROOT / "scripts",
        ROOT / "benchmarks",
    ]
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in sorted(scan_root.rglob("*.py")):
            if path.resolve() == THIS_FILE:
                continue
            tree = ast.parse(
                path.read_text(encoding="utf-8-sig"), filename=str(path)
            )
            imported_legacy_names = {
                alias.asname or alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module == "dtwcpp"
                for alias in node.names
                if alias.name in _LEGACY_CALL_NAMES
            }
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr in _LEGACY_ATTRIBUTE_NAMES
                ):
                    relative = path.relative_to(ROOT).as_posix()
                    violations.add(f"{relative}:{node.lineno}:{node.attr}")
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                elif (
                    isinstance(node.func, ast.Name)
                    and node.func.id in imported_legacy_names
                ):
                    name = node.func.id
                else:
                    continue
                if name in _LEGACY_CALL_NAMES:
                    relative = path.relative_to(ROOT).as_posix()
                    violations.add(f"{relative}:{node.lineno}:{name}")
    return sorted(violations)


def test_ordinary_repository_code_has_no_legacy_calls():
    """Only this focused compatibility fixture may execute legacy callables."""
    global _ORDINARY_HYGIENE

    violations = _ordinary_legacy_calls()
    assert violations == [], "ordinary legacy calls remain:\n" + "\n".join(violations)
    _ORDINARY_HYGIENE = True
