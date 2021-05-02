#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from scipy.linalg import expm


#%% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # Compute nominal updates
    new_p = p + (v * dt) + 0.5 * ((q.as_matrix() @ (a_m - a_b)) + g) * (dt ** 2)
    new_v = v + ((q.as_matrix() @ (a_m - a_b)) + g) * dt
    new_q = q * Rotation.from_matrix(expm(skew(np.squeeze(w_m - w_b)) * dt))

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # Compute noise terms
    Fi = np.zeros((18, 12))
    rows = np.arange(3, 15)
    cols = np.arange(0, 12)
    Fi[rows, cols] = 1

    Qi = np.zeros((12, 12))
    Vi = (accelerometer_noise_density ** 2) * (dt ** 2) * np.eye(3)
    Theta_i = (gyroscope_noise_density ** 2) * (dt ** 2) * np.eye(3)
    Ai = (accelerometer_random_walk ** 2) * dt * np.eye(3)
    Omega_i = (gyroscope_random_walk ** 2) * dt * np.eye(3)
    Qi[0:3, 0:3] = Vi
    Qi[3:6, 3:6] = Theta_i
    Qi[6:9, 6:9] = Ai
    Qi[9:12, 9:12] = Omega_i
    noise = Fi @ Qi @ Fi.T

    # Update covariance matrix
    Fx = np.zeros((18, 18))
    Fx[0:3, 0:3] = np.eye(3)
    Fx[0:3, 3:6] = np.eye(3) * dt
    Fx[3:6, 3:6] = np.eye(3)
    Fx[3:6, 6:9] = -q.as_matrix() @ skew(np.squeeze(a_m - a_b)) * dt
    Fx[3:6, 9:12] = -q.as_matrix() * dt
    Fx[3:6, 15:18] = np.eye(3) * dt
    Fx[6:9, 6:9] = (Rotation.from_rotvec(np.squeeze(w_m - w_b) * dt).as_matrix()).T
    Fx[6:9, 12:15] = -np.eye(3) * dt
    Fx[9:12, 9:12] = np.eye(3)
    Fx[12:15, 12:15] = np.eye(3)
    Fx[15:18, 15:18] = np.eye(3)

    # Return updated 18x18 covariance matrix with noise term added
    return Fx @ error_state_covariance @ Fx.T + noise


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # Compute innovation vector
    Pc = q.as_matrix().T @ (Pw - p)
    Xc = Pc[0, 0]
    Yc = Pc[1, 0]
    Zc = Pc[2, 0]
    innovation = uv - np.array([[Xc / Zc], [Yc / Zc]])

    # Check if measurement is an inlier
    if norm(innovation) > error_threshold:
        return (p, v, q, a_b, w_b, g), error_state_covariance, innovation

    # Compute Jacobian
    Ht = np.zeros((2, 18))
    u = uv[0, 0]
    v = uv[1, 0]
    del_zt_pc = (1 / Zc) * np.array([[1, 0, -u], [0, 1, -v]])
    Ht[:, 0:3] = del_zt_pc @ -q.as_matrix().T
    Ht[:, 6:9] = del_zt_pc @ skew(np.squeeze(Pc))

    # Compute EKF update
    Kt = error_state_covariance @ Ht.T @ np.linalg.inv(Ht @ error_state_covariance @ Ht.T + Q)
    ekf_update = Kt @ innovation

    # Update nominal state
    p = p + ekf_update[0:3]
    v = v + ekf_update[3:6]
    q = q * Rotation.from_matrix(expm(skew(np.squeeze(ekf_update[6:9]))))
    a_b = a_b + ekf_update[9:12]
    w_b = w_b +ekf_update[12:15]
    g = g + ekf_update[15:18]

    # Update error state covariance matrix
    error_state_covariance = (np.eye(18) - Kt @ Ht) @ error_state_covariance @ (np.eye(18) - Kt @ Ht).T + (Kt @ Q @ Kt.T)


    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation

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