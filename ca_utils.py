import numpy as np


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
