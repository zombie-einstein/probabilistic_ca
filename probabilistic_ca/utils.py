import typing
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

OFFSET = 1e-14


def number_to_base(n: int, *, base: int, width: int) -> np.array:
    """
    Convert a number into it's representation in argument base and
    fixed width

    Args:
        n (int): Number to convert
        base (int): Base to represent number in
        width (int): Width of presentation (padding with 0s)

    Returns:
        np.array: Array of digits
    """
    if n > (base**width) - 1:
        raise ValueError(
            (
                f"{n} is outside the allotted width {width} "
                "of the representation in base {base}"
            )
        )
    ret = np.zeros(width).astype("int")
    idx = 0
    while n:
        ret[idx] = int(n % base)
        n //= base
        idx += 1
    return ret


def rule_arr(
    n,
    base=2,
    perturbations: np.array = None,
    log_prob: bool = True,
    offset: float = OFFSET,
) -> np.array:
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
        np.array: 2-D array representing the CA rule
    """
    n_states = base**3
    out_shape = (n_states, base)

    perturbations = jnp.zeros(out_shape) if perturbations is None else perturbations
    assert perturbations.shape == out_shape, f"Noise should have shape {out_shape}"

    r = number_to_base(n, base=base, width=n_states)

    rp = np.zeros((n_states, base))
    rp[np.arange(n_states), r] = 1.0
    rp = rp + perturbations

    if log_prob:
        rp = np.clip(rp, offset, 1.0 - offset)
        norm = np.sum(rp, axis=1)
        rp = rp / norm[:, np.newaxis]
    else:
        norm = np.sum(rp, axis=1)
        rp = rp / norm[:, np.newaxis]

    return rp


@partial(jax.jit, static_argnames=("log_prob",))
def rule_to_joint(
    r_arr: typing.Union[np.ndarray, jnp.ndarray], log_prob: bool = True
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
        [number_to_base(i, base=n_states, width=4) for i in range(n_states**4)]
    )

    idxs_2 = jnp.array(
        [number_to_base(i, base=n_states, width=2)[::-1] for i in range(n_states**2)]
    )

    pows = (n_states ** np.arange(3))[np.newaxis]

    if log_prob:
        r_arr = r_arr - jax.scipy.special.logsumexp(r_arr, axis=1)[:, jnp.newaxis]
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


def state_to_joint(
    s0: np.ndarray,
    log_prob=True,
    offset: float = OFFSET,
) -> np.ndarray:
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
        np.ndarray: 3d array of joint probabilities.
    """
    w = s0.shape[1]
    n = s0.shape[0]

    if log_prob:
        s0 = np.clip(s0, offset, 1.0 - offset)

    p = s0 / np.sum(s0, axis=0)

    ps = p.take(np.arange(1, w + 1), mode="wrap", axis=1)
    p0 = np.zeros((n, n, w))

    for i in range(n):
        for j in range(n):
            p0[i, j] = p[i] * ps[j]

    if log_prob:
        return np.log(p0)
    else:
        return p0
