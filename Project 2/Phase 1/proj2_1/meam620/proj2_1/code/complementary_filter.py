# %% Imports

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from scipy.linalg import expm


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

    '''
    >>> Construct quaternion multiply together to obtain estimate
    >>> Using measured acceleration vector compute erroor meeasure by looking at magnnitude of accelration vectoor,
    >>> compare to 1 or 9.8
    >>> Compute gain alpha based on output. If nonn-zero, compute correction matrix
    >>> Apply rotation corection using gain alpha
    
    '''

    # Accelaration due to gravity and alpha slope
    g = 9.81
    gain_slope = -10

    # Use linear ODE model to current rotation as function of dt and angular velocity
    curr_rotation = Rotation.from_matrix(expm(skew(angular_velocity) * dt))

    # Do Rotation compoisition to obtain current rotation estimate
    rot_estimate = initial_rotation * curr_rotation

    # Computer error magnitude of acceleration vector
    error_measured = np.abs(np.linalg.norm(linear_acceleration) - g)

    # Compute g_prime and normalize to g
    g_prime = rot_estimate.as_matrix() @ linear_acceleration
    g_prime = g_prime / np.linalg.norm(g)

    # Construct quaternion correction
    # real_corect = np.sqrt((1 + (g * g_prime[0]))/ (2))
    # imag_correct = np.array([0, (g_prime[2])/(g * np.sqrt(2 * (1 + (g * g_prime[0])))),
    #                          (-g_prime[1])/(g * np.sqrt(2 * (1 + (g * g_prime[0]))))])
    # quat_correct = np.append(imag_correct, real_corect)

    real_corect = np.sqrt((1 + (1 * g_prime[0])) / (2))
    imag_correct = np.array([0, (g_prime[2]) / (1 * np.sqrt(2 * (1 + (1 * g_prime[0])))),
                             (-g_prime[1]) / (1 * np.sqrt(2 * (1 + (1 * g_prime[0]))))])
    quat_correct = np.append(imag_correct, real_corect)



    # Compute alpha
    if error_measured > 0.2:
        alpha = 0
    elif error_measured < 0.1:
        alpha = 1
    else:
        alpha = gain_slope * error_measured + 2

    # Construct blended quaternion correction and normalize
    null_rotation = np.array([0, 0, 0, 1])
    quat_correct_prime = (1 - alpha) * null_rotation + (alpha) * quat_correct
    quat_correct_prime = quat_correct_prime / np.linalg.norm(quat_correct_prime)

    # Perform correction
    rot_correction = Rotation.from_quat([quat_correct_prime[0], quat_correct_prime[1],
                                         quat_correct_prime[2], quat_correct_prime[3]])

    return rot_correction * rot_estimate

def skew(v):
    """
        This function computes the skew symmetric representation of a (3, ) vector

        Parameters:
            v,    (3, ) numpy array represennting i,j,k counterpart of quaternion

        Return: 3x3 numpy array

    """

    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
