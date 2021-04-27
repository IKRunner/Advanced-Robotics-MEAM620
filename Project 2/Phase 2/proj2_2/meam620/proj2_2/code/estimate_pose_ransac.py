# Imports

import numpy as np
from scipy.spatial.transform import Rotation


# %%

def estimate_pose(uvd1, uvd2, pose_iterations, ransac_iterations, ransac_threshold):
    """
    Estimate Pose by repeatedly calling ransac

    :param uvd1:
    :param uvd2:
    :param pose_iterations:
    :param ransac_iterations:
    :param ransac_threshold:
    :return: Rotation, R; Translation, T; inliers, array of n booleans
    """

    R = Rotation.identity()

    for i in range(0, pose_iterations):
        w, t, inliers = ransac_pose(uvd1, uvd2, R, ransac_iterations, ransac_threshold)
        R = Rotation.from_rotvec(w.ravel()) * R

    return R, t, inliers

def solve_w_t(uvd1, uvd2, R0):
    """
    solve_w_t core routine used to compute best fit w and t given a set of stereo correspondences

    :param uvd1: 3xn ndarray : normailzed stereo results from frame 1
    :param uvd2: 3xn ndarray : normailzed stereo results from frame 2
    :param R0: Rotation type - base rotation estimate
    :return: w, t : 3x1 ndarray estimate for rotation vector, 3x1 ndarray estimate for translation
    """
    '''
    # initialize inliers to all false
    best_inliers = np.zeros(n, dtype=bool)
    n = uvd1.shape[1]
    If ransanc_iterations < 1:


    A -> 2n * 6
    B -> 2n * 1
    Y = R.as_matrix() @ np.vstack((uvd2[0:2, :], np.ones((1, n))))
    Y1 = Y[0, :]


    '''

    # TODO Your code here replace the dummy return value with a value you compute
    # Generate y matrix
    _, n = uvd1.shape
    Y = R0.as_matrix() @ np.vstack((uvd2[0:2, :], np.ones((1, n))))
    assert Y.shape[0] == 3 and Y.shape[1] == n

    # Loop through all correspondences
    b = np.zeros((2 * n,1))
    A = np.zeros((2 * n, 6))
    for i in range(n):
        # Generate b
        u1_prime = uvd1[0, i]
        v1_prime = uvd1[1, i]
        d2_prime = uvd2[2, i]
        k = np.hstack((np.eye(2,2), np.array([[-u1_prime], [-v1_prime]])))
        b[2*i:(2*i)+2, :] = -k @ Y[:, i][...,None]

        # Generate A matrix
        A[2*i:(2*i)+2, :] = k @ np.array([[0, Y[2, i], -Y[1, i], d2_prime, 0, 0],
                                  [-Y[2, i], 0, Y[0, i], 0, d2_prime, 0],
                                  [Y[1, i], -Y[0, i], 0, 0, 0, d2_prime]])
    assert b.shape[0] == A.shape[0] and b.shape[1] == 1 and A.shape[1] == 6 and b.shape[0] == 2*n

    # Solve system
    x, _, _, _ = np.linalg.lstsq(A,b, rcond=-1)
    assert x.shape[0] == 6 and x.shape[1] == 1

    # Extract rotation and translation
    w = x[0:3]
    t = x[3:6]
    return w, t


def find_inliers(w, t, uvd1, uvd2, R0, threshold):
    """

    find_inliers core routine used to detect which correspondences are inliers

    :param w: ndarray with 3 entries angular velocity vector in radians/sec
    :param t: ndarray with 3 entries, translation vector
    :param uvd1: 3xn ndarray : normailzed stereo results from frame 1
    :param uvd2:  3xn ndarray : normailzed stereo results from frame 2
    :param R0: Rotation type - base rotation estimate
    :param threshold: Threshold to use
    :return: ndarray with n boolean entries : Only True for correspondences that pass the test
    """


    n = uvd1.shape[1]

    # TODO Your code here replace the dummy return value with a value you compute
    return np.zeros(n, dtype='bool')


def ransac_pose(uvd1, uvd2, R0, ransac_iterations, ransac_threshold):
    """

    ransac_pose routine used to estimate pose from stereo correspondences

    :param uvd1: 3xn ndarray : normailzed stereo results from frame 1
    :param uvd2: 3xn ndarray : normailzed stereo results from frame 1
    :param R0: Rotation type - base rotation estimate
    :param ransac_iterations: Number of RANSAC iterations to perform
    :ransac_threshold: Threshold to apply to determine correspondence inliers
    :return: w, t : 3x1 ndarray estimate for rotation vector, 3x1 ndarray estimate for translation
    :return: ndarray with n boolean entries : Only True for correspondences that are inliers

    """
    n = uvd1.shape[1]


    # TODO Your code here replace the dummy return value with a value you compute
    w = t = np.zeros((3,1))
    return w, t, np.zeros(n, dtype='bool')
