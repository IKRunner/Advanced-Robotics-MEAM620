import numpy as np
from scipy.spatial.transform import Rotation as R


class SE3Control(object):
    """

    """

    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """
        # Quadrotor physical parameters.
        self.mass = quad_params['mass']  # kg
        self.Ixx = quad_params['Ixx']  # kg*m^2
        self.Iyy = quad_params['Iyy']  # kg*m^2
        self.Izz = quad_params['Izz']  # kg*m^2
        self.arm_length = quad_params['arm_length']  # meters
        self.rotor_speed_min = quad_params['rotor_speed_min']  # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max']  # rad/s
        self.k_thrust = quad_params['k_thrust']  # N/(rad/s)**2
        self.k_drag = quad_params['k_drag']  # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))  # kg*m^2
        self.g = 9.81  # m/s^2
        self.gamma = self.k_drag / self.k_thrust

        # Initialize kp, kd gains here
        self.kp_pos = np.zeros((3,))
        self.kp_att = np.zeros((3,))
        self.kd_pos = np.zeros((3,))
        self.kd_att = np.zeros((3,))

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # Position gains (x, y, z)
        self.kp_pos = np.array([16.9, 16.9, 81])
        self.kd_pos = np.array([5.6, 5.6, 10.3])

        # Attitude gains (Roll, Pitch, Yaw)
        self.kp_att = np.array([650, 650, 750])
        self.kd_att = np.array([51, 51, 59])
        # print(self.kd_att)

        # Position controller
        rddot_des = flat_output['x_ddot'] - self.kd_pos * (state['v'] - flat_output['x_dot']) - self.kp_pos * \
                    (state['x'] - flat_output['x'])
        # print('Current state z-pos is: ' + str(state['x'][2]))
        # print('Current desired z-pos is: ' + str(flat_output['x'][2]))
        # print('Difference is: ' + str(rddot_des[2]))

        # Compute thrust force and clip
        # cmd_thrust = np.clip(self.mass * (rddot_des[2] + self.g), self.k_thrust * self.rotor_speed_min ** 2,
        #                      self.k_thrust * self.rotor_speed_max ** 2)

        cmd_thrust = self.mass * (rddot_des[2] + self.g)
        # print('thrust: ' + str(cmd_thrust))

        # Compute thet^des and phi^des
        phi_des = (1 / self.g) * (rddot_des[0] * np.sin(flat_output['yaw']) - rddot_des[1] * np.cos(flat_output['yaw']))
        thet_des = (1 / self.g) * (rddot_des[0] * np.cos(flat_output['yaw']) + rddot_des[1] * np.sin(flat_output['yaw']))
        psi_des = flat_output['yaw']

        # print('theta desired: ' + str(thet_des))
        # print('psi desired: ' + str(psi_des))

        # Compute thrust moment
        r = R.from_quat(state['q'])
        # print('quaterion: ' + str(state['q']))
        angles = r.as_euler('ZXY')
        # print('state euler angles: ' + str(angles))
        pqr_dot = np.array([-self.kp_att[0] * (angles[1] - phi_des) - self.kd_att[0] * (state['w'][0] - 0),
                            -self.kp_att[1] * (angles[2] - thet_des) - self.kd_att[1] * (state['w'][1] - 0),
                            -self.kp_att[2] * (angles[0] - psi_des) - self.kd_att[2] * (state['w'][2] - flat_output['yaw'])])
        cmd_moment = self.inertia @ pqr_dot

        # print('psi desired: ' + str(psi_des))
        # print('current psi: ' + str(angles[0]))
        # print('psi difference is: ' + str(angles[0] - psi_des))
        # print('--------------------------------------------------------------------')
        # print('phi desired: ' + str(phi_des))
        # print('current phi: ' + str(angles[1]))
        # print('phi difference is: ' + str(angles[1] - phi_des))
        # print('Moment: ' + str(cmd_moment))

        # Compute propeller speeds
        b = np.insert(cmd_moment, 0, cmd_thrust)
        A = np.array([[1, 1, 1, 1],
                      [0, self.arm_length, 0, -self.arm_length],
                      [-self.arm_length, 0, self.arm_length, 0],
                      [self.gamma, -self.gamma, self.gamma, -self.gamma]])
        F = np.linalg.solve(A, b)
        F[F < 0] = 0
        # F[F < 0] = -F[F < 0]
        cmd_motor_speeds = np.sqrt(F / self.k_thrust)

        # Derive quaternion
        cmd_q = R.from_euler('ZXY', [flat_output['yaw'], phi_des, thet_des]).as_quat()
        # print('Propeller speeds: ' + str(cmd_motor_speeds))

        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment,
                         'cmd_q': cmd_q}
        return control_input