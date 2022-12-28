import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

import typing

import ca_utils as ca


@partial(jax.jit, static_argnames=("log_prob",))
def rule_to_joint(
    r_arr: typing.Union[np.ndarray, jnp.ndarray],
    log_prob: bool = True
) -> jnp.ndarray:
    """
    Convert rule array to joint probability array.

    Convert a 2d array representing a probabilistic
    CA rule into a 3d array representing the joint
    probability of CA states given the preceding state.

    Args:
        r_arr: 2d probabilistic rule array.
        log_prob: If ``True`` ``r_arr`` will be treated as
            an array of log probabilities.

    Returns:
        jnp.ndarray: 2d array of joint probabilities
            of state pairs from preceding states.
    """
    n_states = r_arr.shape[1]

    idxs_4 = jnp.array(
        [
            ca.number_to_base(i, base=n_states, width=4)
            for i in range(n_states ** 4)
        ]
    )

    idxs_2 = jnp.array(
        [
            ca.number_to_base(i, base=n_states, width=2)[::-1]
            for i in range(n_states ** 2)
        ]
    )

    pows = (n_states ** np.arange(3))[np.newaxis]

    if log_prob:
        r_arr = (
            r_arr -
            jax.scipy.special.logsumexp(r_arr, axis=1)[:, jnp.newaxis]
        )
    else:
        r_arr = r_arr / jnp.sum(r_arr, axis=1)[:, jnp.newaxis]

    def inner_unroll(i):
        s1, s2 = i
        ln = jnp.sum(idxs_4[:, :3] * pows, axis=1)
        rn = jnp.sum(idxs_4[:, 1:] * pows, axis=1)

        if log_prob:
            return r_arr.at[ln, s1].get() + r_arr.at[rn, s2].get()
        else:
            return r_arr.at[ln, s1].get() * r_arr.at[rn, s2].get()

    return jax.vmap(inner_unroll)(idxs_2).reshape(n_states, n_states, -1)


def state_to_joint(s0: np.ndarray) -> np.ndarray:
    """
    Convert an initial state to joint probability array.

    Convert a 2d probabilistic initial state array to
    a 3d array of joint probabilities of adjacent
    state of the initial state.

    Args:
        s0: 2d array of probabilistic initial state.

    Returns:
        np.ndarray: 3d array of joint probabilities.
    """
    w = s0.shape[1]
    n = s0.shape[0]

    p = s0 / np.sum(s0, axis=0)
    ps = p.take(np.arange(1, w + 1), mode='wrap', axis=1)

    p0 = np.zeros((n, n, w))

    for i in range(n):
        for j in range(n):
            p0[i, j] = p[i] * ps[j]

    return p0


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
        [
            ca.number_to_base(i, base=n_states, width=4)
            for i in range(n_states ** 4)
        ]
    )

    idxs_2 = np.array(
        [
            ca.number_to_base(i, base=n_states, width=2)[::-1]
            for i in range(n_states ** 2)
        ]
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

            return jnp.where(pd != 0, pl * pr * pn / pd, 0.)

    if log_prob:
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
        [
            ca.number_to_base(i, base=n_states, width=2)[::-1]
            for i in range(n_states ** 2)
        ]
    )

    if log_prob:
        def inner_mutual_info(i):
            s1, s2 = i
            pd1 = jax.scipy.special.logsumexp(probs[:, s1], axis=1)
            pd2 = jax.scipy.special.logsumexp(probs[:, s2], axis=1)
            pd2 = pd2.take(jnp.arange(1, w + 1), mode="wrap", axis=1)

            return (
                    jnp.exp(probs[:, s1, s2]) *
                    (probs[:, s1, s2] - (pd1 + pd2))
            )

    else:

        def inner_mutual_info(i):
            s1, s2 = i
            pd1 = jnp.sum(probs[:, s1], axis=1)
            pd2 = jnp.sum(probs[:, s2], axis=1)
            pd2 = pd2.take(jnp.arange(1, w + 1), mode="wrap", axis=1)

            return (
                    probs[:, s1, s2] *
                    jnp.log(probs[:, s1, s2] / (pd1 * pd2))
            )

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
        return jax.scipy.special.logsumexp(probs, axis=1)

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
