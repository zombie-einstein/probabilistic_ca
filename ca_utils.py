import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import minmax_scale


def number_to_base(n: int, *, base: int, width: int) -> np.array:
    """
    Convert a number into it's representation in argument weight and
    fixed width

    Args:
        n (int): Number to convert
        base (int): Base to represent number in
        width (int): Width of presentation (padding with 0s)

    Returns:
        np.array: Array of digits
    """
    if n > (base ** width) - 1:
        raise ValueError(f"{n} is outside the allotted width {width} of the representation in base {base}")
    ret = np.zeros(width).astype('int')
    idx = 0
    while n:
        ret[idx] = int(n % base)
        n //= base
        idx += 1
    return ret


def base_to_number(n, *, base):
    """Convert number in base array back to an integer value"""
    return np.sum(n * (base ** np.arange(len(n))))


def rule_arr(n, idxs=None, perbs=None):
    """
    Generate an array representing a ca-rule with possible deviations from that
    rule to create probabilistic update rules

    Args:
        n (int): Rule number to use as a base rule
        idxs (list): List of indices to apply perturbations
        perbs (list): List of perturbations corresponding to the indices list

    Returns:
        np.array: 2-D array representing the CA rule
    """
    idxs = idxs or ()
    perbs = perbs or ()

    assert len(idxs) == len(perbs), "Index and perturbation lists must be the same length"

    r = number_to_base(n, base=2, width=8).astype('float')

    for j, k in zip(idxs, perbs):
        r[j] = r[j] - k if r[j] > 0 else r[j] + k

    rp = np.zeros((8, 2))
    rp[:, 1] = r
    rp[:, 0] = 1 - rp[:, 1]

    return rp


def joint_probability_map(r_array):
    """
    Convert a rule array into conditional mapping from rules for joint
    probabilities

    Args:
        r_array (np.array): Update rule array to be converted

    Returns:
        np.array: 3d array representing the conditional update rules
    """
    n_states = r_array.shape[1]
    ret = np.zeros((n_states, n_states, n_states ** 4, 5))

    for i in range(0, n_states ** 2):

        k = number_to_base(i, base=n_states, width=2)

        for j in range(n_states ** 4):
            nb = number_to_base(j, base=n_states, width=4)
            a = base_to_number(nb[:3], base=n_states)
            b = base_to_number(nb[1:], base=n_states)
            ret[k[0], k[1], j, :4] = nb
            ret[k[0], k[1], j, 4] = r_array[a][k[0]] * r_array[b][k[1]]

    return ret


def model_runner(rule: np.array, steps: int, initial_state: np.array):
    """
    Run the model for the argument initial state, update rule and number of
    steps. Currently only supports 2 states.

    Args:
        rule (np.array): Update rule array
        steps (int): Number of steps to run the model for
        initial_state (np.array): Initial state represented as a 1d array

    Returns:
        np.array: 4d array of joint probability distributions
    """
    # Create rule mapping
    rule_map = joint_probability_map(rule)

    # Width of the array
    width = len(initial_state)

    # TODO: Extend to more than 2 states
    n_states = 2

    # Initial State
    s0 = np.array(initial_state)

    # Vector p(i)
    p0 = np.zeros((width, n_states), dtype='float64')

    # Initialize probabilities from the initial state
    for i, j in enumerate(s0):
        p0[i, 1] = j
        p0[i, 0] = 1-j

    # Shift probabilities array to the right
    shift_p0 = p0.take(np.arange(1, width + 1), mode='wrap', axis=0)

    # Initialize empty joint probability array with steps and width
    joint = np.zeros((steps, width, n_states, n_states), dtype='float64')

    # And then get the joint (independent) probabilities
    joint[0, :, 0, 0] = p0[:, 0] * shift_p0[:, 0]
    joint[0, :, 1, 0] = p0[:, 1] * shift_p0[:, 0]
    joint[0, :, 0, 1] = p0[:, 0] * shift_p0[:, 1]
    joint[0, :, 1, 1] = p0[:, 1] * shift_p0[:, 1]

    # Update function called each step
    def update(x, y, l_arr, c_arr, r_arr, p_arr0, p_arr1):
        r_map = rule_map[x][y][:, :4].astype('int')
        r_prob = rule_map[x][y][:, 4:]
        den = [p_arr0[:, r[1]] * p_arr1[:, r[2]] for r in r_map]
        num = [
            l_arr[:, r[0], r[1]] * c_arr[:, r[1], r[2]] * r_arr[:, r[2], r[3]]
            for r in r_map]
        slices = [np.divide(n, d, out=np.zeros_like(n), where=d != 0) for n, d
                  in zip(num, den)]
        slices = np.multiply(r_prob, slices)
        return np.stack(slices).sum(axis=0)

    # Update each row in turn from previous row
    for i in range(1, joint.shape[0]):
        joint_t = joint[i - 1]

        # Left and right shifts
        shift_lt = joint_t.take(np.arange(-1, width - 1), mode='wrap', axis=0)
        shift_rt = joint_t.take(np.arange(1, width + 1), mode='wrap', axis=0)

        # Per-site marginal probabilities
        probs = joint_t.sum(axis=2)

        # Shifted marginals
        probs_r = probs.take(np.arange(1, width + 1), mode='wrap', axis=0)

        # Update each of the joint probabilities
        joint[i, :, 0, 0] = update(0, 0, shift_lt, joint_t, shift_rt, probs,
                                   probs_r)
        joint[i, :, 1, 0] = update(1, 0, shift_lt, joint_t, shift_rt, probs,
                                   probs_r)
        joint[i, :, 0, 1] = update(0, 1, shift_lt, joint_t, shift_rt, probs,
                                   probs_r)
        joint[i, :, 1, 1] = update(1, 1, shift_lt, joint_t, shift_rt, probs,
                                   probs_r)

    return joint


def mutual_info(arr):
    """
    Calculate the mutual information of a joint probability array

    Args:
        arr (np.array): 4d joint probability array

    Returns:
        np.array: 2d mutual information array
    """
    p0 = np.sum(arr, axis=3)
    p1 = np.sum(arr, axis=2)

    def sub_mut(i, j):
        m = p0[:, :, i] * p1[:, :, j]
        d = np.log(m, out=np.zeros_like(m), where=m != 0)
        l = np.log(arr[:, :, i, j],
                   out=np.zeros_like(arr[:, :, i, j]),
                   where=arr[:, :, i, j] != 0)
        return np.multiply(arr[:, :, i, j], l - d)

    m00 = sub_mut(0, 0)
    m10 = sub_mut(1, 0)
    m01 = sub_mut(0, 1)
    m11 = sub_mut(1, 1)

    return m00 + m10 + m01 + m11


def flat_joint_entropy(arr):
    """
    Calculate the entropy of the flattened joint probability array

    Args:
        arr (np.array): 4d joint probability array

    Returns:
        np.array: 2d mutual information array
    """
    return entropy(arr.reshape(arr.shape[0], arr.shape[1], -1), axis=2)


def min_max_scale_rows(arr):
    """Min-scale across rows of a 2-d array"""
    return minmax_scale(arr, axis=1)


def checks(joint_prob_arr):
    """
    Checks that the model is correctly producing probability distributions
    """

    # Check that marginal probabilities are the same when
    # summed from left-to-right or right-to-left
    last_row = joint_prob_arr[-1]
    assert np.isclose(last_row.sum(axis=1),
                      last_row.sum(axis=2).take(np.arange(1, last_row.shape[0] + 1),
                                                mode='wrap', axis=0)).all()

    np.isclose(1, np.sum(joint_prob_arr) / (joint_prob_arr.shape[0]*joint_prob_arr.shape[1]))

    print("Ok")
