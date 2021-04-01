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


# Set A-matrix for matrix solution
def set_matrix(n_coeff, n_seg, n, time):
    # Initialize A-matrices
    Ax = np.zeros((n_coeff * n_seg, n_coeff * n_seg))
    Ay = np.zeros((n_coeff * n_seg, n_coeff * n_seg))
    Az = np.zeros((n_coeff * n_seg, n_coeff * n_seg))
    #################################################################
    # Initialize constraints for start boundary conditions
    x_start_const = np.zeros((n_coeff, n_coeff))
    y_start_const = np.zeros((n_coeff, n_coeff))
    z_start_const = np.zeros((n_coeff, n_coeff))

    # Populate values for start boundary conditions
    x_start_const[0, -1] = 1
    x_start_const[1, -1 - 1] = 1
    y_start_const[0, -1] = 1
    y_start_const[1, -1 - 1] = 1
    z_start_const[0, -1] = 1
    z_start_const[1, -1 - 1] = 1

    # Spline is minimum jerk or higher
    if n > 2:
        size = n - 2
        terms = np.multiply.accumulate(np.arange(size) + 2)

        # Rows to place constants
        rows = (np.arange(size) + 2)[np.newaxis, ...].T

        # Columns to place constants
        cols = (n_coeff - 1) - rows

        # Indices to place constants
        idx = np.concatenate((rows, cols), axis=1)

        # Place constants
        x_start_const[idx[:, 0], idx[:, 1]] = terms[:]
        y_start_const[idx[:, 0], idx[:, 1]] = terms[:]
        z_start_const[idx[:, 0], idx[:, 1]] = terms[:]

    # Add start boundary constraints to A matrix
    Ax[0:n_coeff, 0:n_coeff] = x_start_const
    Ay[0:n_coeff, 0:n_coeff] = y_start_const
    Az[0:n_coeff, 0:n_coeff] = z_start_const

    # Initialize constraints for end boundary conditions
    x_end_const = np.zeros((n_coeff, n_coeff))
    y_end_const = np.zeros((n_coeff, n_coeff))
    z_end_const = np.zeros((n_coeff, n_coeff))

    # There will be n row entries in this part
    # Figure out way to make non-dimensional problem TODO

    # Populate values for position end boundary constraints
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    x_exp = x ** np.linspace(n_coeff - 1, 0, n_coeff)
    y_exp = y ** np.linspace(n_coeff - 1, 0, n_coeff)
    z_exp = z ** np.linspace(n_coeff - 1, 0, n_coeff)
    x_fpos = lambdify(x, x_exp, 'numpy')
    y_fpos = lambdify(y, y_exp, 'numpy')
    z_fpos = lambdify(z, z_exp, 'numpy')
    x_pos_val = np.array(x_fpos(time[-1, 0]))
    y_pos_val = np.array(y_fpos(time[-1, 0]))
    z_pos_val = np.array(z_fpos(time[-1, 0]))

    # Populate derivative values for end boundary constraints
    x_pos_deriv = np.zeros((n - 1, n_coeff))
    y_pos_deriv = np.zeros((n - 1, n_coeff))
    z_pos_deriv = np.zeros((n - 1, n_coeff))

    # First derivatives
    x_deriv = Array(x_exp).diff(x)
    y_deriv = Array(y_exp).diff(y)
    z_deriv = Array(z_exp).diff(z)
    for i in range(n - 1):
        x_fderiv = lambdify(x, x_deriv, 'numpy')
        y_fderiv = lambdify(y, y_deriv, 'numpy')
        z_fderiv = lambdify(z, z_deriv, 'numpy')
        x_pos_deriv[i, :] = np.array(x_fderiv(time[-1, 0]))
        y_pos_deriv[i, :] = np.array(y_fderiv(time[-1, 0]))
        z_pos_deriv[i, :] = np.array(z_fderiv(time[-1, 0]))

        # Compute subsequent derivatives
        x_deriv = Array(x_deriv).diff(x)
        y_deriv = Array(y_deriv).diff(y)
        z_deriv = Array(z_deriv).diff(z)

    # Populate values for end boundary conditions
    x_end_const[n_coeff - n:n_coeff, :] = np.vstack((x_pos_val, x_pos_deriv))
    y_end_const[n_coeff - n:n_coeff, :] = np.vstack((y_pos_val, y_pos_deriv))
    z_end_const[n_coeff - n:n_coeff, :] = np.vstack((z_pos_val, z_pos_deriv))

    # Add end boundary constraints to A matrix
    Ax[0:n_coeff, (n_coeff * n_seg) - n_coeff:(n_coeff * n_seg)] = x_end_const
    Ay[0:n_coeff, (n_coeff * n_seg) - n_coeff:(n_coeff * n_seg)] = y_end_const
    Az[0:n_coeff, (n_coeff * n_seg) - n_coeff:(n_coeff * n_seg)] = z_end_const

    #################################################################
    # Initialize intermediate position constraints
    num_pos_constr = 2 * (n_seg - 1)
    x_pos_const = np.zeros((num_pos_constr, n_coeff * n_seg))
    y_pos_const = np.zeros((num_pos_constr, n_coeff * n_seg))
    z_pos_const = np.zeros((num_pos_constr, n_coeff * n_seg))

    # Extract first n_seg - 1 time segments and apply to length n_coeff time polynomial
    x_time = np.array(x_fpos(time[0:-1, 0])).T
    y_time = np.array(y_fpos(time[0:-1, 0])).T
    z_time = np.array(z_fpos(time[0:-1, 0])).T

    # Add ones to position constraint matrix
    num_pos_constr = 2 * (n_seg - 1)
    rows = np.linspace(2, num_pos_constr, n_seg - 1, dtype=np.int32) - 1
    cols = np.linspace(2 * n_coeff, n_coeff * n_seg, n_seg - 1, dtype=np.int32) - 1
    idx = np.concatenate((rows[..., np.newaxis], cols[..., np.newaxis]), axis=1)
    x_pos_const[idx[:, 0], idx[:, 1]] = 1
    y_pos_const[idx[:, 0], idx[:, 1]] = 1
    z_pos_const[idx[:, 0], idx[:, 1]] = 1

    # Add time polynomials to position constraint matrix
    rng = np.arange(0, (n_coeff * n_seg) - n_coeff, 1)
    sub_cols = np.row_stack(np.array_split(rng, n_seg - 1))
    rows = np.arange(0, num_pos_constr, 2)
    x_pos_const[rows[..., np.newaxis], sub_cols] = x_time
    y_pos_const[rows[..., np.newaxis], sub_cols] = y_time
    z_pos_const[rows[..., np.newaxis], sub_cols] = z_time

    # Add intermediate position constraints to A matrix
    Ax[n_coeff:(n_coeff + num_pos_constr), :] = x_pos_const
    Ay[n_coeff:(n_coeff + num_pos_constr), :] = y_pos_const
    Az[n_coeff:(n_coeff + num_pos_constr), :] = z_pos_const

    #################################################################
    # Initialize continuity constraints
    num_cont_constr = (2 * (n - 1)) * (n_seg - 1)
    x_cont_const = np.zeros((num_cont_constr, n_coeff * n_seg))
    y_cont_const = np.zeros((num_cont_constr, n_coeff * n_seg))
    z_cont_const = np.zeros((num_cont_constr, n_coeff * n_seg))

    # Initialize constants
    size = 2 * (n - 1)
    terms = -np.multiply.accumulate(np.arange(size) + 1)

    # Rows to place constants
    rows = (np.arange(0, num_cont_constr))[np.newaxis, ...].T

    # Columns to place constants
    start = (n_coeff * n_seg) - 2
    end = (n_coeff * n_seg) - (size + 1)
    rng = start - end
    cols = np.arange(start, start - (((rng + 1) * (n_seg - 1)) + (2 * (n_seg - 1))), -1).reshape(-1, (size + 2))
    cols = cols[:, :size]
    cols = cols[::-1].ravel()[...,np.newaxis]

    '''
    m = 4 # No. segments
    k = 3
    num = 22
    x = np.arange(num, num - (((k + 1) * (m - 1)) + (2 * (m - 1))), -1).reshape(-1, (k + 1+ 2))
    x = x[:, :k + 1]
    # Flip with 
    x = x[::-1].ravel()[...,np.newaxis]

    '''

    # Place constants
    x_cont_const[rows, cols] = np.tile(terms, n_seg - 1)[...,np.newaxis]
    y_cont_const[rows, cols] = np.tile(terms, n_seg - 1)[...,np.newaxis]
    z_cont_const[rows, cols] = np.tile(terms, n_seg - 1)[...,np.newaxis]

    # Compute derivatives
    x_pos_deriv = np.zeros((size, n_coeff * (n_seg - 1)))
    y_pos_deriv = np.zeros((size, n_coeff * (n_seg - 1)))
    z_pos_deriv = np.zeros((size, n_coeff * (n_seg - 1)))

    # First derivatives
    x_deriv = Array(x_exp).diff(x)
    y_deriv = Array(y_exp).diff(y)
    z_deriv = Array(z_exp).diff(z)

    for i in range(size):
        x_fderiv = lambdify(x, x_deriv, 'numpy')
        y_fderiv = lambdify(y, y_deriv, 'numpy')
        z_fderiv = lambdify(z, z_deriv, 'numpy')

        #################
        # Print current derivative
        # print(x_fderiv.__doc__)
        ##################

        # Access relevant elements
        result = np.array(x_fderiv(time[0:-1, 0]), dtype=object)
        x_curr_deriv = np.stack(np.array(x_fderiv(time[0:-1, 0]), dtype=object)[:-1 - 1 - i]).T
        y_curr_deriv = np.stack(np.array(y_fderiv(time[0:-1, 0]), dtype=object)[:-1 - 1 - i]).T
        z_curr_deriv = np.stack(np.array(z_fderiv(time[0:-1, 0]), dtype=object)[:-1 - 1 - i]).T

        # Add back constants
        x_curr_deriv = np.hstack((x_curr_deriv, np.repeat(-terms[i], n_seg - 1)[..., None]))
        y_curr_deriv = np.hstack((y_curr_deriv, np.repeat(-terms[i], n_seg - 1)[..., None]))
        z_curr_deriv = np.hstack((z_curr_deriv, np.repeat(-terms[i], n_seg - 1)[..., None]))

        # Add (i + 0) zeros to columns
        x_curr_deriv = np.hstack((x_curr_deriv, np.zeros((n_seg - 1, i + 1))))
        y_curr_deriv = np.hstack((y_curr_deriv, np.zeros((n_seg - 1, i + 1))))
        z_curr_deriv = np.hstack((z_curr_deriv, np.zeros((n_seg - 1, i + 1))))

        # Store current derivative
        x_pos_deriv[i, :] = x_curr_deriv.ravel()
        y_pos_deriv[i, :] = y_curr_deriv.ravel()
        z_pos_deriv[i, :] = z_curr_deriv.ravel()

        # Compute subsequent derivatives
        x_deriv = Array(x_deriv).diff(x)
        y_deriv = Array(y_deriv).diff(y)
        z_deriv = Array(z_deriv).diff(z)

    # Stack derivatives for each time segment
    x_pos_deriv = np.vstack(np.hsplit(x_pos_deriv, int(n_coeff * (n_seg - 1) / n_coeff)))
    y_pos_deriv = np.vstack(np.hsplit(y_pos_deriv, int(n_coeff * (n_seg - 1) / n_coeff)))
    z_pos_deriv = np.vstack(np.hsplit(z_pos_deriv, int(n_coeff * (n_seg - 1) / n_coeff)))

    # Compute columns
    rng = np.arange(0, n_coeff * (n_seg - 1), 1)
    sub_cols = np.row_stack(np.array_split(rng, n_coeff * (n_seg - 1)/n_coeff))
    sub_cols = np.repeat(sub_cols, size, axis=0)

    # Add derivative polynomials to continuity constraint matrix
    x_cont_const[rows, sub_cols] = x_pos_deriv
    y_cont_const[rows, sub_cols] = y_pos_deriv
    z_cont_const[rows, sub_cols] = z_pos_deriv

    # Add continuity constraints to A matrix
    Ax[(n_coeff * n_seg) - num_cont_constr:, :] = x_cont_const
    Ay[(n_coeff * n_seg) - num_cont_constr:, :] = y_cont_const
    Az[(n_coeff * n_seg) - num_cont_constr:, :] = z_cont_const

    return np.array([Ax, Ay, Az])