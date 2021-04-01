import numpy as np
from sympy import *

"""
        This function......

        Parameters:
            n_coeff,    .....
            n_seg,      .....
            n,          .....
            time,

"""


def time_polynomials(n, n_coeff):
    # Compute time polynomial
    t = Symbol('t')
    t_exp = t ** np.linspace(n_coeff - 1, 0, n_coeff)

    # Store position polynomial in array
    polys = np.zeros((n, n_coeff), dtype=object)
    polys[0, :] = t_exp
    # x_fderiv = lambdify(t, t_exp, 'numpy')

    # First derivative
    polys[1, :] = Array(t_exp).diff(t)

    # Compute subsequent polynomial derivatives
    for i in range(2, n):
        # Subsequent derivatives
        polys[i, :] = Array(polys[i - 1, :]).diff(t)

    # Time and derivative polynomials as function of t
    return lambdify(t, polys, 'numpy')