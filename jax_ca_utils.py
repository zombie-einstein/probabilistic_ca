import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

import ca_utils as ca


def rule_to_joint(r_arr):
    ns = r_arr.shape[1]

    result = np.zeros((ns, ns, ns ** 4))

    for i in range(ns):
        for j in range(ns):
            for k in range(ns ** 4):
                b_arr = ca.number_to_base(k, base=ns, width=4)
                ln = ca.base_to_number(b_arr[:3], base=ns)
                rn = ca.base_to_number(b_arr[1:], base=ns)

                p = r_arr[ln, i] * r_arr[rn, j]

                result[i, j, k] = p

    return result


def state_to_joint(s0):
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
def run_model(rule_joint, p0, n_steps, log_prob=False):

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

            def mutual_info(i):
                s1, s2 = i
                pd1 = jax.scipy.special.logsumexp(p[s1], axis=0)
                pd2 = jax.scipy.special.logsumexp(p[s2], axis=0)
                pd2 = pd2.take(jnp.arange(1, w + 1), mode="wrap", axis=0)

                return jnp.exp(p[s1, s2]) * (p[s1, s2] - (pd1 + pd2))

            mi = jax.vmap(mutual_info)(idxs_2)
            mi = jnp.sum(mi, axis=0)

            ps = jax.scipy.special.logsumexp(p, axis=1)

            return (p, r), (p, ps, mi)
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

            def mutual_info(i):
                s1, s2 = i
                pd1 = jnp.sum(p[s1], axis=0)
                pd2 = jnp.sum(p[s2], axis=0)
                pd2 = pd2.take(jnp.arange(1, w + 1), mode="wrap", axis=0)

                return p[s1, s2] * jnp.log(p[s1, s2] / (pd1 * pd2))

            mi = jax.vmap(mutual_info)(idxs_2)
            mi = jnp.sum(mi, axis=0)

            ps = jnp.sum(p, axis=1)

            return (p, r), (p, ps, mi)

    _, result = jax.lax.scan(step, (p0, rule_joint), None, length=n_steps)

    return result
