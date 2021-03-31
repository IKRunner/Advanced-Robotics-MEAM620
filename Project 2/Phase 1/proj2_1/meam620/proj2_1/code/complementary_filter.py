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

    # Use linear ODE model to compute time position rotation matrix and form to quaternnion
    curr_quaternion = Rotation.from_matrix(expm(skew(angular_velocity) * dt)).as_quat()

    # real = np.cos(np.linalg.norm(angular_velocity * dt) / 2)
    # imag = np.sin(np.linalg.norm(angular_velocity * dt) / 2) * angular_velocity
    # curr_quaternion = np.append(imag, real)

    # Do quaternion multiplication to update rotation matrix
    prev_quaternion = initial_rotation.as_quat()
    real_estimate = prev_quaternion[-1] * curr_quaternion[-1] - prev_quaternion[0:3] @ curr_quaternion[0:3]
    imag_estimate = prev_quaternion[-1] * curr_quaternion[0:3] + curr_quaternion[-1] * prev_quaternion[0:3] \
                    + skew(prev_quaternion[0:3]) @ curr_quaternion[0:3]
    quat_estimate = np.append(imag_estimate, real_estimate)
    rot_estimate = Rotation.from_quat([quat_estimate[0], quat_estimate[1], quat_estimate[2],
                                       quat_estimate[3]]).as_matrix()

    # Compute g_prime and normalize
    g_prime = rot_estimate @ linear_acceleration
    g_prime = g_prime / np.linalg.norm(g_prime)

    # Computer error magnitude of acceleration vector
    error_measured = np.abs(np.linalg.norm(linear_acceleration) - g)
    # print(error_measured)

    # Construct quaternion correction
    imag_correct = np.array([np.sqrt((g_prime[2] + 1) / 2), g_prime[1] / np.sqrt(2 * (g_prime[2] + 1)),
                             -g_prime[0] / np.sqrt(2 * (g_prime[2] + 1))])

    quat_correct = np.append(imag_correct, 0)

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
                                         quat_correct_prime[2], quat_correct_prime[3]]).as_matrix()
    rot_correction * rot_estimate




     # Do quaternion multiplication to update rotation matrix

    # Update rotation using measured angular velocity (How to factor initial condiiton??)
    # omega_hat = np.array([[0, -angular_velocity[2], angular_velocity[1]],
    #                       [angular_velocity[2], 0, -angular_velocity[0]],
    #                       [-angular_velocity[1], angular_velocity[0], 0]])
    #
    # r_estimate = initial_rotation.as_matrix() * np.exp(omega_hat * dt)



    # # Convert current rotation matrix
    # Rotation.from_matrix(rot_estimate * rot_correction)

    # Initial output
    # Rotation.identity()

    return Rotation.from_matrix(rot_correction * rot_estimate)

# rot_estimate = (quat_estimate[0] ** 2 - quat_estimate[1:] @ quat_estimate[1:]) * Rotation.identity().as_matrix() + \
#                2 * quat_estimate[0] * skew(quat_estimate[1:]) +  2 * quat_estimate[1:][:, None] @ quat_estimate[1:][None, :]

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

