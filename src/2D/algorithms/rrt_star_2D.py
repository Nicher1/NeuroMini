import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.spatial
import math
import random
import argparse
import time

from utils.generator import generate_map
from utils.plotter import *

# Node class representing a state in the space
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0

# Multi-Agent RRT* (CMN-RRT*)
class RRTStar:
    def __init__(self, start_positions, goal_positions, num_agents, num_obstacles, map_size, map_type="empty", step_size=1.0, max_iter=500, live_plot=False):
        
        # Map properties
        self.map_size = map_size
        self.map_type = map_type
        self.obstacles = generate_map(map_type, map_size)
        self.obstacle_type = "wall"
        self.goal_region_radius = 4
        
        # Algorithm properties
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = 4.0
        
        # Initialise agents
        self.num_agents = num_agents
        self.agents = [Node(start_positions[i][0], start_positions[i][1]) for i in range(num_agents)]
        self.goals = [Node(goal_positions[i][0], goal_positions[i][1]) for i in range(num_agents)]
        self.trees = [[self.agents[i]] for i in range(num_agents)]
        self.paths = [None] * num_agents
        self.goal_reached = [False] * num_agents

        # Results per agent
        self.agents_results = {
            i: {
                "iterations": 0,
                "path_length": 0.0,
                "time": 0.0
            } for i in range(num_agents)
        }

        # Visualization setup
        self.live_plot = live_plot
        self.fig, self.ax = plt.subplots()
        setup_visualization(self.ax, self.agents, self.goals, self.map_size, self.obstacle_type, self.obstacles)
            
    # This function gets creates a random node anywhere on the map
    # The idea is to try to move towards that node and explore.
    # This is part of the random exploration part
    def get_weighted_random_node(self, agent_id):
        goal = self.goals[agent_id]
        if random.random() < 0.2:
            return Node(goal.x, goal.y)
        else:
            x = random.uniform(0, self.map_size[0])
            y = random.uniform(0, self.map_size[1])
            if random.random() < 0.7:
                x = (x + goal.x) / 2
                y = (y + goal.y) / 2
            return Node(x, y)
    
    # This function get the nearest node to the random node
    # previously generated. 
    def get_nearest_node(self, tree, rand_node):
        points = np.array([[node.x, node.y] for node in tree])
        tree_kdtree = scipy.spatial.KDTree(points)
        _, index = tree_kdtree.query([rand_node.x, rand_node.y])
        return tree[index]
    
    def compute_path_length(self, path):
        length = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            length += math.hypot(dx, dy)
        return length
    
    def plan(self):
        start_time = [time.time() for _ in range(self.num_agents)]

        for iter_count in range(self.max_iter):
            for agent_id in range(self.num_agents):
                if self.goal_reached[agent_id]:
                    continue

                rand_node = self.get_weighted_random_node(agent_id)
                nearest_node = self.get_nearest_node(self.trees[agent_id], rand_node)
                new_node = self.steer(nearest_node, rand_node)

                if self.is_collision_free(nearest_node, new_node):
                    near_nodes = self.find_near_nodes(self.trees[agent_id], new_node)
                    new_node = self.choose_parent(near_nodes, new_node)
                    self.trees[agent_id].append(new_node)
                    self.rewire(self.trees[agent_id], near_nodes, new_node)
                    draw_tree(self.ax, new_node, live_plot=self.live_plot)

                    if self.reached_goal(new_node, self.goals[agent_id]):
                        self.paths[agent_id] = self.generate_final_path(new_node)
                        self.goal_reached[agent_id] = True
                        draw_path(self.ax, self.paths, agent_id)

                    if self.reached_goal(new_node, self.goals[agent_id]):
                        self.paths[agent_id] = self.generate_final_path(new_node)
                        self.goal_reached[agent_id] = True

                        # Save result info
                        duration = time.time() - start_time[agent_id]
                        path_len = self.compute_path_length(self.paths[agent_id])

                        self.agents_results[agent_id]["iterations"] = iter_count
                        self.agents_results[agent_id]["path_length"] = path_len
                        self.agents_results[agent_id]["time"] = duration

                        draw_path(self.ax, self.paths, agent_id)


        print("\n--- Agent Results ---")
        for agent_id, result in self.agents_results.items():
            print(f"Agent {agent_id}:")
            print(f"  Iterations:   {result['iterations']}")
            print(f"  Path Length:  {result['path_length']:.2f}")
            print(f"  Time Taken:   {result['time']:.2f} seconds")
                
    
    def steer(self, from_node, to_node):
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_node = Node(from_node.x + self.step_size * math.cos(theta),
                        from_node.y + self.step_size * math.sin(theta))
        new_node.parent = from_node
        return new_node
    
    def ccw(self, A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def segments_intersect(self, A, B, C, D):
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def is_collision_free(self, from_node, to_node):
        # For each wall
        for (x_min, y_min, width, height) in self.obstacles:
            x_max = x_min + width
            y_max = y_min + height

            # Break wall into 4 edges as line segments
            wall_edges = [
                ((x_min, y_min), (x_max, y_min)),  # bottom
                ((x_max, y_min), (x_max, y_max)),  # right
                ((x_max, y_max), (x_min, y_max)),  # top
                ((x_min, y_max), (x_min, y_min)),  # left
            ]

            # Line segment from agent's path
            p1 = (from_node.x, from_node.y)
            p2 = (to_node.x, to_node.y)

            # Check for intersection with any wall edge
            for (w1, w2) in wall_edges:
                if self.segments_intersect(p1, p2, w1, w2):
                    return False  # collision

        return True  # no collision
    
    def reached_goal(self, node, goal):
        return np.linalg.norm([node.x - goal.x, node.y - goal.y]) < self.goal_region_radius
    
    def generate_final_path(self, goal_node):
        path = []
        node = goal_node
        while node is not None:
            path.append([node.x, node.y])
            node = node.parent
        return path[::-1]
    
    def find_near_nodes(self, tree, new_node, radius=10.0):
        near_nodes = []
        for node in tree:
            dist = math.hypot(node.x - new_node.x, node.y - new_node.y)
            if dist <= radius:
                near_nodes.append(node)
        return near_nodes
    
    def choose_parent(self, near_nodes, new_node):
        best_parent = None
        best_cost = float('inf')

        for node in near_nodes:
            temp_node = self.steer(node, new_node)
            if self.is_collision_free(node, temp_node):
                cost = node.cost + self.distance(node, temp_node)
                if cost < best_cost:
                    best_parent = node
                    best_cost = cost

        if best_parent:
            new_node.parent = best_parent
            new_node.cost = best_cost
        return new_node
    
    def distance(self, node1, node2):
        return math.hypot(node1.x - node2.x, node1.y - node2.y)

    def rewire(self, tree, near_nodes, new_node):
        for node in near_nodes:
            new_cost = new_node.cost + self.distance(new_node, node)
            if new_cost < node.cost and self.is_collision_free(new_node, node):
                node.parent = new_node
                node.cost = new_cost

    def animate(self):
        plt.show()
    
# Execution

