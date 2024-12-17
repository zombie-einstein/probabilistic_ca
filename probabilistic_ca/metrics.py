from functools import partial

import chex
import jax
import jax.numpy as jnp
from jax.experimental import checkify

from .utils import number_to_base, permutations


@partial(jax.jit, static_argnames=("log_prob",))
def mutual_information(
    probs: jnp.ndarray,
    log_prob: bool = True,
) -> jnp.ndarray:
    """
    Get mutual information from joint probabilities.

    Args:
        probs: 4d array of joint probability time series.
        log_prob: If ``True`` then values will be treated
            as log probabilities.

    Returns:
        jnp.ndarray: Mutual information time-series.
    """

    n_states = probs.shape[1]
    w = probs.shape[3]
    f_perms = checkify.checkify(permutations)
    errs, idxs_2 = f_perms(n_states, 2)
    idxs_2 = jnp.flip(idxs_2, axis=1)

    if log_prob:

        def inner_mutual_info(i):
            s1, s2 = i
            pd1 = jax.nn.logsumexp(probs[:, s1], axis=1)
            pd2 = jax.nn.logsumexp(probs[:, s2], axis=1)
            pd2 = pd2.take(jnp.arange(1, w + 1), mode="wrap", axis=1)
            norm = pd1 + pd2
            p = probs[:, s1, s2]
            return jnp.exp(p) * (p - norm)

    else:

        def inner_mutual_info(i):
            s1, s2 = i
            pd1 = jnp.sum(probs[:, s1], axis=1)
            pd2 = jnp.sum(probs[:, s2], axis=1)
            pd2 = pd2.take(jnp.arange(1, w + 1), mode="wrap", axis=1)
            norm = pd1 * pd2
            p = probs[:, s1, s2]
            return jnp.where(p > 0.0, p * jnp.log(p / norm), 0.0)

    mi = jax.vmap(inner_mutual_info)(idxs_2)
    return jnp.sum(mi, axis=0)


@partial(jax.jit, static_argnames=("log_prob",))
def state_probabilities(
    probs: jnp.ndarray,
    log_prob: bool = True,
) -> jnp.ndarray:
    """
    Get individual state probabilities from joint probabilities.

    Args:
        probs: 4d array of joint probability time series.
        log_prob: If ``True`` then values will be treated
            as log probabilities.

    Returns:
        jnp.ndarray: State probability time-series.
    """

    if log_prob:
        return jax.nn.logsumexp(probs, axis=1)

    else:
        return jnp.sum(probs, axis=1)


@partial(jax.jit, static_argnames=("log_prob",))
def entropy(
    probs: jnp.ndarray,
    log_prob: bool = True,
) -> jnp.ndarray:
    """
    Get entropy series from joint probabilities.

    Args:
        probs: 4d array of joint probability time series.
        log_prob: If ``True`` then values will be treated
            as log probabilities.

    Returns:
        jnp.ndarray: Entropy time-series.
    """

    if log_prob:
        return jnp.sum(jnp.exp(probs) * probs, axis=(1, 2))

    else:
        x = jnp.where(probs > 0.0, jnp.log(probs) * probs, 0.0)
        return jnp.sum(x, axis=(1, 2))


@partial(jax.jit, static_argnames=("log_prob",))
def state_joint_probabilities(x: chex.Array, log_prob: bool) -> chex.Array:
    """
    Recover the probability of a discrete state from joint probs

    Args:
        x: Array of joint probabilities (i.e. the state of the
            model inside a step)
        log_prob: Whether x represents log probabilities.

    Returns:
        Array of probabilities for each permutation of discrete
        states.
    """
    n = x.shape[2]
    s = x.shape[0]
    idx = jnp.arange(n)

    f = checkify.checkify(partial(number_to_base, base=s, width=n))

    if log_prob:

        def inner(i: chex.Numeric) -> chex.Numeric:
            err, idx_a = f(i)
            idx_b = idx_a.take(idx + 1, mode="wrap")

            marginals = x[idx_a, :, idx]
            marginals = jax.nn.logsumexp(marginals, axis=1)
            probs = x[idx_a, idx_b, idx]
            prob_state = jnp.sum(probs) - jnp.sum(marginals)
            return prob_state

    else:

        def inner(i: chex.Numeric) -> chex.Numeric:
            err, idx_a = f(i)
            idx_b = idx_a.take(idx + 1, mode="wrap")

            marginals = x[idx_a, :, idx]
            marginals = jnp.sum(marginals, axis=1)
            probs = x[idx_a, idx_b, idx]
            prob_state = jnp.prod(probs) / jnp.prod(marginals)
            return prob_state

    return jax.vmap(inner)(jnp.arange(s**n))
