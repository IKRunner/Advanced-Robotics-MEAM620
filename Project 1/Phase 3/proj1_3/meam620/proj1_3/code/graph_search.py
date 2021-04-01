from heapq import heappush, heappop, heapify  # Recommended.
import numpy as np

from flightsim.world import World

from .occupancy_map import OccupancyMap  # Recommended.


def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    # Initialize empty priority queue to store f[v] of neighbors when expanding a node
    Q = []
    Q_lookup = {}

    # Initialize container to store expanded nodes (keys are tuples of coordinates at metric center)
    Y = {}

    # Set algorithim to A-star or Dikjstra
    def heuristic(x, dest_node, curr_node):
        return np.linalg.norm(dest_node - curr_node) if x else 0

    # Initialize  heuristic and cost to go
    # Convert goal to index then convert to metric center to compute distance
    h = heuristic(astar, occ_map.index_to_metric_center(goal_index), occ_map.index_to_metric_center(start_index))

    # Initialize start node, push to queue
    node_start = Node(0, h, tuple(occ_map.index_to_metric_center(start_index)), None)
    heappush(Q, node_start)
    Q_lookup[node_start.index] = node_start

    # Continuously loop while goal node is not in closed list and Queue is not empty
    o = 0
    while tuple(occ_map.index_to_metric_center(goal_index)) not in Y.keys() and Q != []:
        o += 1
        u = heappop(Q)
        u.is_closed = True
        try:
            del Q_lookup[u.index]
        except KeyError:
            print("pass number is: " + str(o))
        Y[tuple(u.index)] = u

        # Convert current u to index coordinates; container for 26 connect
        vox_idx = occ_map.metric_to_index(u.index)
        valid_neighbors = np.zeros((26, 3))

        # Extract neighbouring voxels that are valid
        i = 0
        for z in range(vox_idx[2] - 1, vox_idx[2] + 2):
            for y in range(vox_idx[1] - 1, vox_idx[1] + 2):
                for x in range(vox_idx[0] - 1, vox_idx[0] + 2):
                    if x == vox_idx[0] and y == vox_idx[1] and z == vox_idx[2]:
                        continue
                    if occ_map.is_valid_index([x, y, z]) and not occ_map.is_occupied_index([x, y, z]) and \
                            tuple(occ_map.index_to_metric_center([x, y, z])) not in Y.keys():
                        valid_neighbors[i, :] = np.array([x, y, z])
                        i += 1

        # Convert neighbor voxel indices to metric center coordinates
        valid_neighbors = valid_neighbors[0:i]
        valid_neighbors_metric = tuple(occ_map.index_to_metric_center(valid_neighbors))

        # Process neighbors
        for v in valid_neighbors_metric:
            # Cost to come is Euclidean metric
            dist = u.g + np.linalg.norm(v - u.index)

            # Node is already in queue
            if tuple(v) in Q_lookup.keys():
                g_v = Q_lookup[tuple(v)].g

            # Node is not in queue
            if tuple(v) not in Q_lookup.keys():
                g_v = np.inf

            # Update cost value of node if node is already in queue with lower cost to come and update parent
            if dist < g_v and tuple(v) in Q_lookup.keys():
                Q_lookup[tuple(v)].g = dist
                Q_lookup[tuple(v)].parent = u.index

            # Add node to queue if not there already and cost to come is low
            if dist < g_v and tuple(v) not in Q_lookup.keys():
                # Create new node with relevant heuristic and current u as parent and update queue
                h = heuristic(astar, occ_map.index_to_metric_center(goal_index), v)
                new_node = Node(dist, h, tuple(v), u.index)
                heappush(Q, new_node)
                Q_lookup[new_node.index] = new_node

            # Do nothing if the distance is greater than the current cost
            if dist >= g_v:
                continue

    # Construct path
    nodes_expanded = len(Y)

    # Exit if no path found
    if tuple(occ_map.index_to_metric_center(goal_index)) not in Y.keys():
        return None, nodes_expanded

    # Construct path
    path = np.zeros((nodes_expanded, 3))
    i = 0
    result = tuple(occ_map.index_to_metric_center(goal_index))
    path[i, :] = result
    i += 1
    while Y[result].parent is not None:
        result = Y[result].parent
        path[i, :] = result
        i += 1
    path = path[0:i]

    # Prepend and append true goal and start to path and flip path
    path = np.vstack((np.vstack((goal, path)), start))[::-1]

    # Return a tuple (path, nodes_expanded)
    return path, nodes_expanded


class Node:
    """
    Container for node data.Nodes are sortable by the value of f.This is just an example to get you started!
    """

    def __init__(self, g, h, index, parent):
        self.f = g + h
        self.g = g
        self.h = h
        self.index = index
        self.parent = parent
        self.is_closed = False

    def __lt__(self, other):
        """
        Return True if this node comes before other node in priority.
        Note that you are free to choose how you break ties in the f value.
        """
        return self.f < other.f

    def __repr__(self):
        """
        Define a pretty display of the node information for debugging.
        """
        return f"Node g={self.g}, h={self.h}, index={self.index}, parent={self.parent}, is_closed={self.is_closed}"