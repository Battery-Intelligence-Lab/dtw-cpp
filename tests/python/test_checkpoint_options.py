"""Problem.checkpoint parity with C++ automatic mid-fill checkpointing.

Mirrors ``dtwc::Problem::checkpoint`` (dtwc/Problem.hpp) and the validation
messages raised by ``Problem::fill_distance_matrix()``.
"""

import numpy as np
import pytest

import dtwcpp


_SERIES = [[0.0, 0.0], [1.0, 1.0], [4.0, 4.0], [9.0, 9.0]]


def _problem(name="ckpt"):
    prob = dtwcpp.Problem(name)
    prob.set_data(_SERIES, [str(i) for i in range(len(_SERIES))])
    return prob


def _generation_count(directory):
    generations = directory / "generations"
    if not generations.is_dir():
        return 0
    return sum(1 for entry in generations.iterdir() if entry.is_dir())


class TestCheckpointMember:
    def test_in_place_mutation_survives(self):
        """Without rv_policy::reference_internal the getter returns a copy."""
        prob = _problem()
        prob.checkpoint.enabled = True
        prob.checkpoint.save_interval = 3
        prob.checkpoint.directory = "somewhere"
        assert prob.checkpoint.enabled is True
        assert prob.checkpoint.save_interval == 3
        assert prob.checkpoint.directory == "somewhere"

    def test_whole_struct_assignment(self):
        prob = _problem()
        options = dtwcpp.CheckpointOptions()
        options.enabled = True
        options.save_interval = 7
        options.directory = "elsewhere"
        prob.checkpoint = options
        assert prob.checkpoint.enabled is True
        assert prob.checkpoint.save_interval == 7
        assert prob.checkpoint.directory == "elsewhere"

    def test_defaults_match_cpp(self):
        prob = _problem()
        assert prob.checkpoint.enabled is False
        assert prob.checkpoint.save_interval == 100
        assert prob.checkpoint.directory == "./checkpoints"


class TestAutomaticMidFill:
    def test_row_blocks_publish_one_retained_generation_and_resume(self, tmp_path):
        """Mirrors C++ unit_test_checkpoint case 9.

        Each row block publishes a generation that supersedes the previous
        one, so a completed fill leaves exactly one, holding the full matrix.
        """
        directory = tmp_path / "ckpt"
        reference = _problem()
        reference.fill_distance_matrix()

        prob = _problem()
        prob.checkpoint.enabled = True
        prob.checkpoint.save_interval = 1
        prob.checkpoint.directory = str(directory)
        prob.fill_distance_matrix()

        assert prob.is_distance_matrix_filled()
        assert _generation_count(directory) == 1

        restored = _problem()
        assert dtwcpp.load_checkpoint(restored, str(directory)) is True
        assert restored.is_distance_matrix_filled()
        np.testing.assert_array_equal(restored.distance_matrix(),
                                      reference.distance_matrix())

    def test_resumed_fill_keeps_the_restored_matrix(self, tmp_path):
        """fill_distance_matrix() must not wipe a loaded checkpoint."""
        directory = tmp_path / "ckpt"
        source = _problem()
        source.checkpoint.enabled = True
        source.checkpoint.save_interval = 2
        source.checkpoint.directory = str(directory)
        source.fill_distance_matrix()

        restored = _problem()
        assert dtwcpp.load_checkpoint(restored, str(directory)) is True
        before = restored.distance_matrix()
        restored.fill_distance_matrix()
        np.testing.assert_array_equal(restored.distance_matrix(), before)

    def test_disabled_writes_nothing(self, tmp_path):
        directory = tmp_path / "ckpt"
        prob = _problem()
        prob.checkpoint.directory = str(directory)
        prob.fill_distance_matrix()
        assert not directory.exists()


class TestCheckpointValidation:
    def test_non_positive_interval_is_rejected(self, tmp_path):
        prob = _problem()
        prob.checkpoint.enabled = True
        prob.checkpoint.save_interval = 0
        prob.checkpoint.directory = str(tmp_path / "ckpt")
        with pytest.raises(
            dtwcpp.InvalidInput,
            match=r"save_interval must be at least 1 row",
        ):
            prob.fill_distance_matrix()
        assert not (tmp_path / "ckpt").exists()

    def test_empty_directory_is_rejected(self):
        prob = _problem()
        prob.checkpoint.enabled = True
        prob.checkpoint.directory = ""
        with pytest.raises(
            dtwcpp.InvalidInput,
            match=r"requires a non-empty checkpoint\.directory",
        ):
            prob.fill_distance_matrix()
