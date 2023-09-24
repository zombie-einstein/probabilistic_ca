from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .utils import number_to_base


@partial(jax.jit, static_argnames=("n_steps", "log_prob"))
def run_model(
    rule_joint: jnp.ndarray,
    p0: jnp.ndarray,
    n_steps: int,
    log_prob=True,
) -> jnp.ndarray:
    """
    Run CA and return time-series of state and stats.

    Run CA for ``n_steps`` from initial state and joint
    probability rule. Return tuple containing the
    series of joint probabilities, probabilities of individual
    states and mutual information.

    Args:
        rule_joint: Probabilistic CA rule represented
            as joint probability.
        p0: Initial state as joint probability.
        n_steps: Number of steps to run CA for
        log_prob: If ``True`` the update rule and states
            will be treated as log probabilities.

    Returns:
        typing.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            Tuple containing time-series of joint probabilities,
             probabilities of individual states and mutual
             information over execution of the CA.
    """

    w = p0.shape[2]
    n_states = rule_joint.shape[0]

    idxs_4 = np.array(
        [number_to_base(i, base=n_states, width=4) for i in range(n_states**4)]
    )

    idxs_2 = np.array(
        [number_to_base(i, base=n_states, width=2)[::-1] for i in range(n_states**2)]
    )

    if log_prob:

        def p_joint(p, s):
            s0, s1, s2, s3 = s
            pl = p[s0, s1].take(jnp.arange(-1, w - 1), mode="wrap", axis=0)
            pr = p[s2, s3].take(jnp.arange(1, w + 1), mode="wrap", axis=0)
            pn = p[s1, s2]

            pd1 = jax.scipy.special.logsumexp(p[s1], axis=0)
            pd2 = jax.scipy.special.logsumexp(p[s2], axis=0)
            pd2 = pd2.take(jnp.arange(1, w + 1), mode="wrap", axis=0)

            return pl + pr + pn - (pd1 + pd2)

        def step(carry, _):
            p, r = carry

            prev_probs = jax.vmap(p_joint, in_axes=(None, 0))(p, idxs_4)

            def inner_step(i):
                s1, s2 = i
                _p = prev_probs + r[s1, s2][:, jnp.newaxis]
                return jax.scipy.special.logsumexp(_p, axis=0)

            p = jax.vmap(inner_step)(idxs_2)
            p = p.reshape((n_states, n_states, -1))

            return (p, r), p

    else:

        def p_joint(p, s):
            s0, s1, s2, s3 = s
            pl = p[s0, s1].take(jnp.arange(-1, w - 1), mode="wrap", axis=0)
            pr = p[s2, s3].take(jnp.arange(1, w + 1), mode="wrap", axis=0)
            pn = p[s1, s2]

            pd1 = jnp.sum(p[s1], axis=0)
            pd2 = jnp.sum(p[s2], axis=0)
            pd2 = pd2.take(jnp.arange(1, w + 1), mode="wrap", axis=0)
            pd = pd1 * pd2

            return jnp.where(pd != 0, pl * pr * pn / pd, 0.0)

        def step(carry, _):
            p, r = carry

            prev_probs = jax.vmap(p_joint, in_axes=(None, 0))(p, idxs_4)

            def inner_step(i):
                s1, s2 = i
                _p = prev_probs * r[s1, s2][:, jnp.newaxis]
                return jnp.sum(_p, axis=0)

            p = jax.vmap(inner_step)(idxs_2)
            p = p.reshape((n_states, n_states, -1))

            return (p, r), p

    _, result = jax.lax.scan(step, (p0, rule_joint), None, length=n_steps)

    return result
