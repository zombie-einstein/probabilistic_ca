import numpy as np
import pytest

from probabilistic_ca import utils


@pytest.mark.parametrize("log_prob", [True, False])
@pytest.mark.parametrize("base", [2, 3])
def test_rule_arr(log_prob: bool, base: int):
    a1 = utils.rule_arr(128, base=base, log_prob=log_prob)
    assert a1.shape == (base**3, base)
    assert np.allclose(np.sum(a1, axis=1), 1.0)

    noise = np.random.uniform(size=(base**3, base)) * 0.1
    a2 = utils.rule_arr(128, base=base, perturbations=noise, log_prob=log_prob)
    assert a2.shape == (base**3, base)
    assert np.allclose(np.sum(a2, axis=1), 1.0)


@pytest.mark.parametrize("log_prob", [True, False])
@pytest.mark.parametrize("base", [2, 3])
def test_joint_state(log_prob: bool, base: int):
    width = 100

    s0 = np.random.uniform(size=(base, width))
    s0 = utils.state_to_joint(s0, log_prob=log_prob)
    if log_prob:
        s0 = np.exp(s0)
    s_total = np.sum(s0, axis=(0, 1))

    assert s0.shape == (base, base, width)
    assert np.allclose(s_total, 1.0)
