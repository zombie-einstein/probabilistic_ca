import jax
import jax.numpy as jnp
import pytest

import probabilistic_ca as ca


@pytest.mark.parametrize("log_prob", [True, False])
def test_model_run(log_prob: bool) -> None:
    """Just test model runs"""
    width = 10
    steps = 20
    r = ca.rule_arr(35, log_prob=log_prob)
    j = ca.rule_to_joint(r, log_prob=log_prob)
    k = jax.random.PRNGKey(101)
    rand_probs = jax.random.choice(k, jnp.array([0.0, 1.0]), (width,))
    rand_probs = jnp.vstack([rand_probs, 1.0 - rand_probs])
    s0 = ca.state_to_joint(rand_probs, log_prob=log_prob)
    out = ca.run_model(j, s0, steps, log_prob=log_prob, show_progress=False)

    assert out.shape == (steps, 2, 2, width)

    if log_prob:
        out = jnp.exp(out)

    assert jnp.allclose(jnp.sum(out, axis=(1, 2)), 1.0)
