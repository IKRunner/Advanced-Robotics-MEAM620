import numpy as np


"""
        This function......

        Parameters:
            path,     .....

"""

# Function initializes a sparse set of waypoints
def set_sparse(path):
    points = path
    n_pts, _ = path.shape
    # Container that stores indices of points to delete
    to_del = []
    p = 0
    pts = np.zeros((3, 3))

    # Set min and max angle for truncation
    max_ang = 180.0
    min_ang = 125.0

    # Go through all points in dense A-star path
    for i in range(n_pts):
        pts[p, :] = path[i, :]
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
                # Save indices of points to delete
                to_del.append(i - 1)
            p = 0

    # Delete points
    return np.delete(points, to_del, axis=0)
