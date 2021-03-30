# %% Imports

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


# %%

def complementary_filter_update(initial_rotation, angular_velocity, linear_acceleration, dt):
    """
    Implements a complementary filter update

    :param initial_rotation: rotation_estimate at start of update
    :param angular_velocity: angular velocity vector at start of interval in radians per second
    :param linear_acceleration: linear acceleration vector at end of interval in meters per second squared
    :param dt: duration of interval in seconds
    :return: final_rotation - rotation estimate after update
    """

    # TODO Your code here - replace the return value with one you compute

    # Form quaternion
    # Update rotation using measured angular velocity (How to factor initial condiiton??)
    omega_hat = np.array([[0, -angular_velocity[2], angular_velocity[1]],
                          [angular_velocity[2], 0, -angular_velocity[0]],
                          [-angular_velocity[1], angular_velocity[0], 0]])

    r_estimate = initial_rotation.as_matrix() * np.exp(omega_hat * dt)


    # Construct quaternion multiply together to obtain estimate
    # Using measured acceleration vector compute erroor meeasure by looking at magnnitude of accelration vectoor,
    # compare to 1 or 9.8
    # Compute gain alpha based on output. If nonn-zero, compute correction matrix
    # Apply rotation corection using gain alpha


    # Convert current rotation matrix
    t=1
    return Rotation.identity()
