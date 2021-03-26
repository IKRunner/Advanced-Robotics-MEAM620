import numpy as np

"""
        This function......

        Parameters:
            points,     .....
            n_seg,      .....
            n_coeff,    .....
            n,          .....

"""

# Set b vector for matrix solution
def set_b(points, n_seg, n_coeff, n):
    # n_seg = 2  # TO DELETE!!!!!!!!!!!!!!!!!!!!!!

    # Total position constraints
    num_pos_constr = 2 * (n_seg - 1)

    # Total continuitiy constraints
    num_cont_constr = (2 * (n - 1)) * (n_seg - 1)

    # Initialize b vector
    bx = np.zeros((n_coeff + num_pos_constr + num_cont_constr, ))
    by = np.zeros((n_coeff + num_pos_constr + num_cont_constr, ))
    bz = np.zeros((n_coeff + num_pos_constr + num_cont_constr, ))

    # Add position start-boundary constraints
    bx[0] = points[0, 0]
    by[0] = points[0, 1]
    bz[0] = points[0, 2]

    # Add position end-boundary constraints
    bx[int(n_coeff / 2)] = points[-1, 0]
    by[int(n_coeff / 2)] = points[-1, 1]
    bz[int(n_coeff / 2)] = points[-1, 2]

    # Add intermediate position constraints
    bx[n_coeff:n_coeff + num_pos_constr] = np.repeat(points[1:-1, 0], 2)
    by[n_coeff:n_coeff + num_pos_constr] = np.repeat(points[1:-1, 1], 2)
    bz[n_coeff:n_coeff + num_pos_constr] = np.repeat(points[1:-1, 2], 2)

    # Continuity constraints already factored in as zero!
    return np.array([bx, by, bz])