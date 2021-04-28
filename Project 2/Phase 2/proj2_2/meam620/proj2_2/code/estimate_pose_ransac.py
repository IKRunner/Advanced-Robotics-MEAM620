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

    # TODO Your code here replace the dummy return value with a value you compute
    # Generate y matrix
    _, n = uvd1.shape
    Y = R0.as_matrix() @ np.vstack((uvd2[0:2, :], np.ones((1, n))))

    # Loop through all correspondences
    b = np.zeros((2 * n, 1))
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

    # Solve system
    x, _, _, _ = np.linalg.lstsq(A,b, rcond=-1)

    # Extract rotation and translation vectors
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

    # TODO Your code here replace the dummy return value with a value you compute
    n = uvd1.shape[1]
    # Initialize all inliers to false
    inliers = np.zeros((n, ), dtype='bool')

    # Loop through all correspondences
    for i in range(n):
        u1_prime = uvd1[0, i]
        v1_prime = uvd1[1, i]
        u2_prime = uvd2[0, i]
        v2_prime = uvd2[1, i]
        d2_prime = uvd2[2, i]
        k = np.hstack((np.eye(2, 2), np.array([[-u1_prime], [-v1_prime]])))
        rot = (np.eye(3,3) + skew(w)) @ R0.as_matrix()
        u1 = np.array([[u2_prime], [v2_prime], [1]])
        tran = d2_prime * t[...,None]

        # Generate discrepancy vector
        delta = k @ (((rot) @ (u1)) + (tran))
        assert delta.shape[0] == 2 and delta.shape[1] == 1

        # Preserve all inliers
        inliers[i] = np.linalg.norm(delta) < threshold
    return inliers


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
    '''
     # initialize inliers to all false
     best_inliers = np.zeros(n, dtype=bool)
     n = uvd1.shape[1]
     If ransanc_iterations < 1:

     '''
    # TODO Your code here replace the dummy return value with a value you compute
    n = uvd1.shape[1]
    proposed_soln = np.zeros((6, ransac_iterations))
    proposed_inliers = np.zeros((n, ransac_iterations), dtype='bool')

    # All correspondences are inliers for zero iterations
    if ransac_iterations < 1:
        w, t = solve_w_t(uvd1, uvd2, R0)
        return w, t, np.ones((n, ), dtype='bool')

    # Cycle through all k iterations
    for k in range(ransac_iterations):
        # Select three random subsets of correspondences w/o replacement
        subsets = np.random.choice(n, 3)

        # Index into stereo measurements using random subsets
        uvd1_subsets = uvd1[:, subsets]
        uvd2_subsets = uvd2[:, subsets]

        # Generate proposed w and t vector solutions using random subsets
        w, t = solve_w_t(uvd1_subsets, uvd2_subsets, R0)
        proposed_soln[0:3, k] = np.squeeze(w)
        proposed_soln[3:6, k] = np.squeeze(t)

        # Find inliers for proposed vector solutions
        proposed_inliers[:, k] = find_inliers(np.squeeze(w), np.squeeze(t), uvd1, uvd2, R0, ransac_threshold)

    # Column of largest number of inliers and corresponding weights
    max_inlier = np.argmax(np.sum(proposed_inliers, axis=0))
    w = proposed_soln[0:3, max_inlier][...,None]
    t = proposed_soln[3:6, max_inlier][...,None]

    return w, t, proposed_inliers[:, max_inlier]


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