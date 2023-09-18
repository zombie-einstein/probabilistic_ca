import numpy as np
import pytest

from probabilistic_ca import utils


@pytest.mark.parametrize("log_prob", [True, False])
def test_rule_arr(log_prob: bool):
    a1 = utils.rule_arr(123456789, base=3, log_prob=log_prob)
    assert a1.shape == (27, 3)
    assert np.array_equal(np.sum(a1, axis=1), np.ones((27,)))

    noise = np.random.uniform(size=(27, 3)) * 0.1
    a2 = utils.rule_arr(123456789, base=3, perturbations=noise, log_prob=log_prob)
    assert a2.shape == (27, 3)
    assert np.allclose(np.sum(a2, axis=1), np.ones((27,)))
