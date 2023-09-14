import jax.random

from probabilistic_ca import utils
import numpy as np


def test_rule_arr():
    a1 = utils.rule_arr(123456789, base=3)
    assert a1.shape == (27, 3)
    assert np.array_equal(np.sum(a1, axis=1), np.ones((27,)))

    noise = np.random.uniform(size=(27, 3)) * 0.1
    a2 = utils.rule_arr(123456789, base=3, noise=noise)
    assert a2.shape == (27, 3)
    assert np.allclose(np.sum(a2, axis=1), np.ones((27,)))
