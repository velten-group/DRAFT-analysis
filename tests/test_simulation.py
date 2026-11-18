"""Simulation metrics on small hand-made cases."""
import numpy as np

from draft_analysis import simulation as sim


def test_smoothness_of_a_straight_path():
    # five equally spaced stages on a line: adjacent distance 1, largest distance 4
    values = np.arange(5, dtype=float)[:, None] * np.ones((1, 3)) / np.sqrt(3)
    assert np.isclose(sim.smoothness(values), 1 - 1 / 4)
    assert np.isnan(sim.smoothness(np.ones((5, 3))))


def test_tree_pairs_follow_the_parents():
    assert sim.tree_pairs() == [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)]


def test_prior_snr_of_two_blocks_exceeds_shuffles():
    block = np.kron(np.eye(2, dtype=bool), np.ones((20, 10), dtype=bool))
    rng = np.random.default_rng(0)
    noisy = block ^ (rng.random(block.shape) < 0.05)
    assert sim.prior_snr(noisy)["snr"] > 1.5
