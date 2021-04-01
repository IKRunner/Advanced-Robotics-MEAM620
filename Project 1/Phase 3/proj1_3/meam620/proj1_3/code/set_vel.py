import numpy as np

"""
        This function......

        Parameters:
            points,    .....

"""

# Function sets variable velocity
def set_vel(points, up_scale, down_scale):
    new_pts, _ = points.shape
    vel = np.zeros((points[:, 0].size - 1, ))
    vel[:] = 2.522
    p = 0
    idx = 0
    pts = np.zeros((3, 3))

    # Set min and max angle for velocity modulation
    max_ang = 180.0
    min_ang = 90.0

    # Loop through points in sparse trajectory
    while idx < new_pts:
        pts[p, :] = points[idx, :]
        p += 1
        # Three points now in list
        if p == 3:
            # Compute angle of waypoints
            diff1 = pts[0, :] - pts[1, :]
            diff2 = pts[2, :] - pts[1, :]
            v1 = diff1 / np.linalg.norm(diff1)
            v2 = diff2 / np.linalg.norm(diff2)
            ang = np.rad2deg(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

            # Waypoints are approximately straight
            if max_ang >= ang >= min_ang:
                # Set velocities
                vel[idx - 2] = vel[idx - 2] * up_scale
                vel[idx - 1] = vel[idx - 1] * up_scale

            if ang < min_ang:
                vel[idx - 2] = vel[idx - 2] / down_scale
                vel[idx - 1] = vel[idx - 1] / down_scale
            p = 0
            idx -= 1
        idx += 1
    return vel