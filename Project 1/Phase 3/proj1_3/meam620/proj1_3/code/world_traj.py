import numpy as np

from proj1_3.code.time_polynomials import time_polynomials
from proj1_3.code.graph_search import graph_search
from proj1_3.code.set_matrix import set_matrix
from proj1_3.code.set_sparse import set_sparse
from proj1_3.code.set_vel import set_vel
from proj1_3.code.set_b import set_b


import matplotlib.pylab as plt
import scipy.sparse as sparse


class WorldTraj(object):
    """

    """

    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.23, 0.23, 0.23])  # [0.25, 0.25, 0.25] is initial resolution
        self.margin = 0.38  # 0.5 is initial margin

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.

        # self.points = self.path

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE
        self.n_pts, _ = self.path.shape

        # Only two waypoints in path
        if self.n_pts < 3:
            self.points = self.path

            # Save waypoints and unit direction between waypoints to state
            self.I_hat = np.zeros((self.points[:, 0].size - 1, self.points[0, :].size))

            # Compute direction of travel for each segment between waypoints and save to state
            diff = self.points[1:] - self.points[:-1]
            dist = np.linalg.norm(diff, ord=2, axis=1, keepdims=True)
            self.I_hat = diff / dist

            # Set desired velocity vector
            vel = 2.522
            self.vel_des = vel * self.I_hat

            # Time duration (column 0) and start time (column 1) of each segment
            self.T = np.zeros((self.points[:, 0].size - 1, 2))
            self.T[:, 0] = (dist / vel).T
            self.T[1:, 1] = np.cumsum(self.T[0:-1, 0], axis=0)

        # More than two waypoints in path, set sparse trajectory
        else:
            # Set sparse trajectory
            self.points = set_sparse(self.path)
            self.new_pts, _ = self.points.shape

            # Save waypoints and unit direction between waypoints to state
            self.I_hat = np.zeros((self.points[:, 0].size - 1, self.points[0, :].size))

            # Compute direction of travel for each segment between waypoints and save to state
            diff = self.points[1:] - self.points[:-1]
            dist = np.linalg.norm(diff, ord=2, axis=1, keepdims=True)
            self.I_hat = diff / dist

            # Set variable velocity vectors with scaling factors
            vel = set_vel(self.points, 1.0, 1.0) # (upscale, downscale)

            # Desired variable velocity is set
            self.vel_des = self.I_hat * vel[:, np.newaxis]

            # Time duration (column 0) and start time (column 1) of each segment
            self.T = np.zeros((self.points[:, 0].size - 1, 2))
            self.T[:, 0] = (dist / vel[:, np.newaxis]).T
            self.T[1:, 1] = np.cumsum(self.T[0:-1, 0], axis=0)

            print("Length of dense path is: " + str(self.n_pts))
            print("Length of sparse path is: " + str(self.new_pts))

            # Set b-vector for matrix solution: b = np.array([bx, by, bz])
            num_coeff = 8  # Minimum snap trajectory
            n_deriv = int(num_coeff / 2)
            num_segments, _ = self.T.shape
            b = set_b(self.points, num_segments, num_coeff, n_deriv)

            # Set A-matrix for matrix solution: A = np.array([Ax, Ay, Az])
            A = set_matrix(num_coeff, num_segments, n_deriv, self.T)

            # Compute coefficients and reshape to (3 x num_segments x num_coeff)
            self.coeff = np.linalg.solve(A, b).ravel().reshape(3, num_segments, num_coeff)

            # Compute derivative and position polynomials
            self.polynomials = time_polynomials(n_deriv, num_coeff)

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0


        # f_outputs = np.vstack((np.vstack((np.vstack((np.vstack((x, x_dot)), x_ddot)), x_dddot)), x_ddddot))

        # STUDENT CODE HERE
        # Return first set of waypoints with default parameters when only one waypoint is in path
        if self.points.shape[0] == 1:
            x = self.points[-1, :]
            x_dot = np.array([0, 0, 0])

            flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                           'yaw': yaw, 'yaw_dot': yaw_dot}
            return flat_output

        # Time duration is within trajectory
        if t <= np.sum(self.T[:, 0]):
            # Determine current segment
            curr_seg = np.where(np.logical_or(self.T[:, 1] < t, self.T[:, 1] == t))

            # Current segment time
            curr_time = t - self.T[curr_seg[0][-1], 1]

            # Current coefficients for (x, y, z)
            curr_coeff = self.coeff[:,curr_seg[0][-1], :]

            # Evaluate polynomial at segment time and multpliy coefficients
            polys = np.array(self.polynomials(curr_time))[:, None, :] * curr_coeff

            output = np.sum(polys, axis=2)
            row, _ = output.shape
            full_output = np.vstack((output, np.zeros((5 - row, 3))))
            x = full_output[0, :]
            x_dot = full_output[1, :]
            x_ddot = full_output[2, :]
            x_dddot = full_output[3, :]
            x_ddddot = full_output[4, :]


            # ele = np.where(np.logical_or(self.T[:, 1] < t, self.T[:, 1] == t))
            # Set velocity and position state
            # xdot = self.vel_des[ele[0][-1], :]
            # x = self.points[ele[0][-1], :] + self.vel_des[ele[0][-1], :] * (t - self.T[ele[0][-1], 1])

        # Time is greater than duration of full trajectory
        if t > np.sum(self.T[:, 0]):
            # print('test2')
            x = self.points[-1, :]
            x_dot = np.array([0, 0, 0])

        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot}
        return flat_output




