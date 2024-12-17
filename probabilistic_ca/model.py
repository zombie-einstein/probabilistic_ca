from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax_tqdm
from jax.experimental import checkify

from .utils import permutations


def log_p_joint(p: chex.Array, s: chex.Array) -> chex.Array:
    w = p.shape[2]
    s0, s1, s2, s3 = s
    pl = p[s0, s1].take(jnp.arange(-1, w - 1), mode="wrap", axis=0)
    pr = p[s2, s3].take(jnp.arange(1, w + 1), mode="wrap", axis=0)
    pn = p[s1, s2]

    pd1 = jax.nn.logsumexp(p[s1], axis=0)
    pd2 = jax.nn.logsumexp(p[s2], axis=0)
    pd2 = pd2.take(jnp.arange(1, w + 1), mode="wrap", axis=0)

    return pl + pr + pn - (pd1 + pd2)


def p_joint(p: chex.Array, s: chex.Array) -> chex.Array:
    w = p.shape[2]
    s0, s1, s2, s3 = s
    pl = p[s0, s1].take(jnp.arange(-1, w - 1), mode="wrap", axis=0)
    pr = p[s2, s3].take(jnp.arange(1, w + 1), mode="wrap", axis=0)
    pn = p[s1, s2]

    pd1 = jnp.sum(p[s1], axis=0)
    pd2 = jnp.sum(p[s2], axis=0)
    pd2 = pd2.take(jnp.arange(1, w + 1), mode="wrap", axis=0)
    pd = pd1 * pd2

    return jnp.where(pd != 0, pl * pr * pn / pd, 0.0)


@partial(jax.jit, static_argnames=("n_steps", "log_prob", "show_progress"))
def run_model(
    rule_joint: chex.Array,
    p0: chex.Array,
    n_steps: int,
    log_prob: bool = True,
    show_progress: bool = True,
) -> chex.Array:
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
        show_progress: Display a TQDm progress bar

    Returns:
        Array of joint probabilities at each step.
    """
    n_states = rule_joint.shape[0]
    f_perms = checkify.checkify(permutations)

    errs, idxs_4 = f_perms(n_states, 4)
    errs, idxs_2 = f_perms(n_states, 2)
    idxs_2 = jnp.flip(idxs_2, axis=1)

    if log_prob:

        def step(carry, _):
            p, r = carry
            prev_probs = jax.vmap(log_p_joint, in_axes=(None, 0))(p, idxs_4)

            def inner_step(i):
                _p = prev_probs + r[i[0], i[1]][:, jnp.newaxis]
                return jax.nn.logsumexp(_p, axis=0)

            p_next = jax.vmap(inner_step)(idxs_2)
            p_next = p_next.reshape((n_states, n_states, -1))

            return (p_next, r), p

    else:

        def step(carry, _):
            p, r = carry
            prev_probs = jax.vmap(p_joint, in_axes=(None, 0))(p, idxs_4)

            def inner_step(i):
                _p = prev_probs * r[i[0], i[1]][:, jnp.newaxis]
                return jnp.sum(_p, axis=0)

            p_next = jax.vmap(inner_step)(idxs_2)
            p_next = p_next.reshape((n_states, n_states, -1))

            return (p_next, r), p

    if show_progress:
        step = jax_tqdm.scan_tqdm(n_steps)(step)

    _, result = jax.lax.scan(step, (p0, rule_joint), jnp.arange(n_steps))

    return result
