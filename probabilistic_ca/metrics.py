from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .utils import number_to_base


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

    idxs_2 = np.array(
        [number_to_base(i, base=n_states, width=2)[::-1] for i in range(n_states**2)]
    )

    if log_prob:

        def inner_mutual_info(i):
            s1, s2 = i
            pd1 = jax.nn.logsumexp(probs[:, s1], axis=1)
            pd2 = jax.nn.logsumexp(probs[:, s2], axis=1)
            pd2 = pd2.take(jnp.arange(1, w + 1), mode="wrap", axis=1)

            return jnp.exp(probs[:, s1, s2]) * (probs[:, s1, s2] - (pd1 + pd2))

    else:

        def inner_mutual_info(i):
            s1, s2 = i
            pd1 = jnp.sum(probs[:, s1], axis=1)
            pd2 = jnp.sum(probs[:, s2], axis=1)
            pd2 = pd2.take(jnp.arange(1, w + 1), mode="wrap", axis=1)

            return probs[:, s1, s2] * jnp.log(probs[:, s1, s2] / (pd1 * pd2))

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
        return jnp.sum(probs, axis=2)


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
        return jnp.sum(jnp.log(probs) * probs, axis=(1, 2))
