from functools import partial
from typing import Optional

import chex
import jax
import jax.numpy as jnp
from jax.experimental import checkify

OFFSET = 1e-14


def number_to_base(n: int, *, base: int, width: int) -> chex.Array:
    """
    Convert a number into it's representation in argument base and
    fixed width

    Args:
        n (int): Number to convert
        base (int): Base to represent number in
        width (int): Width of presentation (padding with 0s)

    Returns:
        Array of digits
    """

    checkify.check(
        n < (base**width),
        "{n} is outside the allotted width {w}",
        n=jnp.int32(n),
        w=jnp.int32(width),
    )

    def inner(m, _):
        i = m % base
        m //= base
        return m, i

    _, ret = jax.lax.scan(inner, n, None, length=width)

    return ret


def rule_arr(
    n,
    base=2,
    perturbations: Optional[chex.Array] = None,
    log_prob: bool = True,
    offset: float = OFFSET,
) -> chex.Array:
    """
    Generate an array representing a ca-rule with possible deviations from that
    rule to create probabilistic update rules

    Args:
        n (int): Rule number to use as a base rule
        base (int): Rule possible states, default 2
        perturbations (jnp.array): Noise to add as perturbation to rule
        log_prob (bool): If true small perturbations will be added
            to zero values to avoid logarithm issues
        offset (float): Value to offset by if `log_prob=True`

    Returns:
        2-D array representing the CA rule
    """
    n_states = base**3
    out_shape = (n_states, base)

    perturbations = jnp.zeros(out_shape) if perturbations is None else perturbations
    assert (
        perturbations.shape == out_shape
    ), f"Noise should have shape {out_shape} got {perturbations.shape}"

    r = number_to_base(n, base=base, width=n_states)

    rp = jnp.zeros(out_shape)
    rp = rp.at[jnp.arange(n_states), r].set(1.0)
    rp = rp + perturbations

    if log_prob:
        rp = jnp.clip(rp, offset, 1.0 - offset)

    norm = jnp.sum(rp, axis=1)
    rp = rp / norm[:, jnp.newaxis]

    if log_prob:
        rp = jnp.log(rp)

    return rp


@partial(jax.jit, static_argnames=("log_prob",))
def rule_to_joint(r_arr: chex.Array, log_prob: bool = True) -> chex.Array:
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

    f4 = checkify.checkify(partial(number_to_base, base=n_states, width=4))
    f2 = checkify.checkify(partial(number_to_base, base=n_states, width=2))

    errs, idxs_4 = jax.vmap(f4)(jnp.arange(n_states**4))
    errs, idxs_2 = jax.vmap(f2)(jnp.arange(n_states**2))

    idxs_2 = jnp.flip(idxs_2, axis=1)
    pows = (n_states ** jnp.arange(3))[jnp.newaxis]

    if log_prob:
        r_arr = r_arr - jax.nn.logsumexp(r_arr, axis=1)[:, jnp.newaxis]
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


@partial(jax.jit, static_argnames=("log_prob",))
def state_to_joint(
    s0: jnp.ndarray,
    log_prob=True,
    offset: float = OFFSET,
) -> jnp.ndarray:
    """
    Convert an initial state to joint probability array.

    Convert a 2d probabilistic initial state array to
    a 3d array of joint probabilities of adjacent
    state of the initial state.

    Args:
        s0: 2d array of probabilistic initial state.
        log_prob: `True` if using log-probabilities
        offset: Offset applied to log probabilities

    Returns:
        jnp.ndarray: 3d array of joint probabilities.
    """
    w = s0.shape[1]
    n = s0.shape[0]

    if log_prob:
        s0 = jnp.clip(s0, offset, 1.0 - offset)

    p = s0 / jnp.sum(s0, axis=0)

    ps = jnp.take(p, jnp.arange(1, w + 1), mode="wrap", axis=1)

    p0 = jax.vmap(lambda i: jax.vmap(lambda j: p[i] * ps[j])(jnp.arange(n)))(
        jnp.arange(n)
    )

    if log_prob:
        return jnp.log(p0)
    else:
        return p0
