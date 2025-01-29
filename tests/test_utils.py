from typing import List

import chex
import jax
import jax.numpy as jnp
import pytest

from probabilistic_ca import rule_to_joint, utils


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(101)


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0, 0, 0, 0]),
        (1, [1, 0, 0, 0]),
        (7, [1, 1, 1, 0]),
        (8, [0, 0, 0, 1]),
    ],
)
def test_number_to_base(n: int, expected: List[float]):
    arr = utils.number_to_base(n, base=2, width=4)
    expected = jnp.array(expected)
    assert jnp.array_equal(arr, expected)


def test_permutations():
    perms = utils.permutations(2, 2)
    expected = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    assert jnp.array_equal(perms, expected)


@pytest.mark.parametrize("log_prob", [True, False])
@pytest.mark.parametrize("base", [2, 3])
def test_rule_arr(key: chex.PRNGKey, log_prob: bool, base: int):
    a1 = utils.rule_arr(128, base=base, log_prob=log_prob)
    assert a1.shape == (base**3, base)
    if log_prob:
        a1 = jnp.exp(a1)
    assert jnp.allclose(jnp.sum(a1, axis=1), 1.0)

    noise = jax.random.uniform(key, (base**3, base)) * 0.1
    a2 = utils.rule_arr(128, base=base, perturbations=noise, log_prob=log_prob)
    assert a2.shape == (base**3, base)
    if log_prob:
        a2 = jnp.exp(a2)
    assert jnp.allclose(jnp.sum(a2, axis=1), 1.0)


@pytest.mark.parametrize("log_prob", [True, False])
def test_rule_to_joint(key: chex.PRNGKey, log_prob: bool):
    rule = utils.rule_arr(128, base=2, log_prob=log_prob)
    joint = rule_to_joint(rule, log_prob=log_prob)
    assert joint.shape == (2, 2, 16)

    if log_prob:
        joint = jnp.exp(joint)

    assert jnp.allclose(jnp.sum(joint, axis=(0, 1)), 1.0)


@pytest.mark.parametrize("log_prob", [True, False])
@pytest.mark.parametrize("base", [2, 3])
def test_joint_state(key: chex.PRNGKey, log_prob: bool, base: int):
    width = 100

    s0 = jax.random.uniform(key, (base, width))
    s0 = utils.state_to_joint(s0, convert_log_prob=log_prob)

    if log_prob:
        s0 = jnp.exp(s0)

    s_total = jnp.sum(s0, axis=(0, 1))

    assert s0.shape == (base, base, width)
    assert jnp.allclose(s_total, 1.0)
