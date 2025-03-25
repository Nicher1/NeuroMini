import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import math
import random
import argparse

class Node:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.parent, self.cost = None, 0

class CMN_RRTStar:
    def __init__(self, starts, goals, agents, obstacles, size, step=1.0, max_iter=500):
        self.num_agents, self.map_size, self.step_size, self.max_iter = agents, size, step, max_iter
        self.obstacles = self.generate_obstacles(obstacles)
        self.goal_radius, self.search_radius = 2.0, 5.0
        self.agents = [Node(*starts[i]) for i in range(agents)]
        self.goals = [Node(*goals[i]) for i in range(agents)]
        self.trees, self.paths, self.goal_reached = [[self.agents[i]] for i in range(agents)], [None]*agents, [False]*agents
        self.fig = plt.figure(); self.ax = self.fig.add_subplot(111, projection='3d'); self.setup_visualization()
    
    def generate_obstacles(self, num):
        return [(random.uniform(2, self.map_size[0] - 2),
                 random.uniform(2, self.map_size[1] - 2),
                 random.uniform(2, self.map_size[2] - 2),
                 random.uniform(1, 6)) for _ in range(num)]
    
    
    def get_weighted_random_node(self, agent):
        goal = self.goals[agent]
        if random.random() < 0.2: return Node(goal.x, goal.y, goal.z)
        x, y, z = (random.uniform(0, self.map_size[i]) for i in range(3))
        if random.random() < 0.7: x, y, z = (x+goal.x)/2, (y+goal.y)/2, (z+goal.z)/2
        return Node(x, y, z)
    
    def get_nearest_node(self, tree, rand_node):
        points = np.array([[n.x, n.y, n.z] for n in tree])
        _, index = scipy.spatial.KDTree(points).query([rand_node.x, rand_node.y, rand_node.z])
        return tree[index]
    
    def find_near_nodes(self, tree, new_node):
        return [n for n in tree if np.linalg.norm([n.x - new_node.x, n.y - new_node.y, n.z - new_node.z]) < self.search_radius]
    
    def choose_parent(self, near_nodes, new_node):
        best = min((n for n in near_nodes if self.is_collision_free(new_node)),
                   key=lambda n: n.cost + np.linalg.norm([n.x - new_node.x, n.y - new_node.y, n.z - new_node.z]),
                   default=None)
        if best: new_node.cost, new_node.parent = best.cost + np.linalg.norm([best.x - new_node.x, best.y - new_node.y, best.z - new_node.z]), best
        return new_node
    
    def setup_visualization(self):
        self.ax.set_xlim(0, self.map_size[0]); self.ax.set_ylim(0, self.map_size[1]); self.ax.set_zlim(0, self.map_size[2])
        for i in range(self.num_agents):
            self.ax.scatter(self.agents[i].x, self.agents[i].y, self.agents[i].z, c='b')
            self.ax.scatter(self.goals[i].x, self.goals[i].y, self.goals[i].z, c='r')
        for ox, oy, oz, size in self.obstacles:
            self.ax.scatter(ox, oy, oz, c='black', s=size*10)
    
    def plan(self):
        for _ in range(self.max_iter):
            for agent in range(self.num_agents):
                if self.goal_reached[agent]: continue
                rand, near, new = self.get_weighted_random_node(agent), self.get_nearest_node(self.trees[agent], self.get_weighted_random_node(agent)), self.steer(self.get_nearest_node(self.trees[agent], self.get_weighted_random_node(agent)), self.get_weighted_random_node(agent))
                if self.is_collision_free(new):
                    new = self.choose_parent(self.find_near_nodes(self.trees[agent], new), new)
                    self.trees[agent].append(new)
                    self.draw_tree(new)
                    if self.reached_goal(new, self.goals[agent]): self.paths[agent], self.goal_reached[agent] = self.generate_final_path(new), True
    
    def steer(self, from_node, to_node):
        theta, phi = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x), math.atan2(to_node.z - from_node.z, np.hypot(to_node.x - from_node.x, to_node.y - from_node.y))
        return Node(from_node.x + self.step_size * math.cos(theta) * math.cos(phi),
                    from_node.y + self.step_size * math.sin(theta) * math.cos(phi),
                    from_node.z + self.step_size * math.sin(phi))
    
    def is_collision_free(self, node):
        return all((node.x - ox)**2 + (node.y - oy)**2 + (node.z - oz)**2 > size**2 for ox, oy, oz, size in self.obstacles)
    
    def reached_goal(self, node, goal):
        return np.linalg.norm([node.x - goal.x, node.y - goal.y, node.z - goal.z]) < self.goal_radius
    
    def generate_final_path(self, goal_node):
        path, node = [], goal_node
        while node: path.append([node.x, node.y, node.z]); node = node.parent
        return path[::-1]
    
    def draw_tree(self, node):
        if node.parent:
            self.ax.plot([node.x, node.parent.x], [node.y, node.parent.y], [node.z, node.parent.z], '-b')
        plt.pause(0.01)
    
    def animate(self):
        plt.show()
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CMN-RRT* path planner")

    parser.add_argument("--start", nargs="+", type=int, default=[1, 1, 1, 65, 23, 15],
                        help="Start positions flattened. Example for 2 agents: --start 1 1 1 65 23 15")
    parser.add_argument("--goal", nargs="+", type=int, default=[18, 18, 18, 150, 125, 100],
                        help="Goal positions flattened. Example for 2 agents: --goal 18 18 18 150 125 100")
    parser.add_argument("--num_obstacles", type=int, default=120,
                        help="Number of obstacles in the map")
    parser.add_argument("--map_size", nargs=3, type=int, default=[200, 200, 200],
                        help="Size of the map in 3D")

    args = parser.parse_args()

    # Convert flattened start/goal lists to list of positions
    start_positions = [args.start[i:i+3] for i in range(0, len(args.start), 3)]
    goal_positions = [args.goal[i:i+3] for i in range(0, len(args.goal), 3)]

    num_agents = len(start_positions)
    num_obstacles = args.num_obstacles
    map_size = args.map_size

    cmn_rrt_star = CMN_RRTStar(start_positions, goal_positions, num_agents, num_obstacles, map_size)
    cmn_rrt_star.plan()
    cmn_rrt_star.animate()
 
