import jax
import jax.numpy as jnp
import pytest

import probabilistic_ca as ca

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "log_prob",
    [
        True,
    ],
)
def test_metrics(log_prob: bool):
    # TODO: A few of these have numerical issues if not log-prob or 32-bit
    width = 8
    steps = 20

    r = ca.rule_arr(35, log_prob=log_prob)
    j = ca.rule_to_joint(r, log_prob=log_prob)
    k = jax.random.PRNGKey(101)
    rand_probs = jax.random.choice(k, jnp.array([0.0, 1.0]), (width,))
    rand_probs = jnp.vstack([rand_probs, 1.0 - rand_probs])
    s0 = ca.state_to_joint(rand_probs, log_prob=log_prob)
    out = ca.run_model(j, s0, steps, log_prob=log_prob, show_progress=False)

    mi = ca.mutual_information(out, log_prob=log_prob)
    assert mi.shape == (steps, width)

    ent = ca.entropy(out, log_prob=log_prob)
    assert ent.shape == (steps, width)

    probs = ca.state_probabilities(out, log_prob=log_prob)

    if log_prob:
        probs = jnp.exp(probs)

    assert probs.shape == (steps, 2, width)
    assert jnp.allclose(jnp.sum(probs, axis=1), 1.0)

    state_probs = ca.state_joint_probabilities(out[-1], log_prob=log_prob)

    if log_prob:
        state_probs = jnp.exp(state_probs)

    assert state_probs.shape == (2**width,)
    assert jnp.isclose(jnp.sum(state_probs), 1.0)
