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
    def __init__(self, id, agent, x, y):
        self.id = id
        self.agent = agent # 0, 1, 2 ... -1 means no agent (goal node)
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0

# Multi-Agent RRT* (CMN-RRT*)
class CMNRRTStar:
    def __init__(self, start_position, goal_position, num_agents, map_size, map_type="empty", step_size=1.0, max_iter=500, live_plot=False):
        
        # Map properties
        self.map_size = map_size
        self.map_type = map_type
        self.obstacles = generate_map(map_type, map_size)
        self.obstacle_type = "wall"
        self.goal_region_radius = 4

        self.start_node = Node("start", 0, start_position[0], start_position[1])
        self.goal_node = Node("goal", -1, goal_position[0], goal_position[1])


        
        # Algorithm variables
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = 4.0
        
        # Initialise agents
        self.agents = [self.start_node]
        self.num_agents = num_agents
        
        # Agent generation variables
        self.cost_threshold = (self.map_size[0] + self.map_size[1]) / 10
        self.agent_node_radius = 20 # Helps checking collisions for agent placement

        self.generate_agents()

        self.trees = [[self.agents[i]] for i in range(self.num_agents)]
        self.paths = [None] * self.num_agents
        self.goal_reached = [False] * self.num_agents

        # Results per agent
        self.agents_results = {
            i: {
                "iterations": 0,
                "path_length": 0.0,
                "time": 0.0
            } for i in range(self.num_agents)
        }

        # Visualization setup
        self.live_plot = live_plot
        self.fig, self.ax = plt.subplots()
        setup_visualization(self.ax, self.agents, self.goal_node, self.map_size, self.obstacle_type, self.obstacles)


    # This function generates self.num_agents. These agents are actually
    # initial nodes for CMN_RRT. If the node can't be placed due to collision
    # the function will try to find another random position
    # A distance is calculated between the node and the starting node and goal node
    # and based on this the node is gonna be used or not
    def generate_agents(self):

        # Starts from 1 because Agent 0 is created in the starting node
        for i in range(1, self.num_agents - 1):
            x = random.uniform(0, self.map_size[0])
            y = random.uniform(0, self.map_size[1])

            # Repeat until the agent is placed on the map
            while not self.is_agent_collision_free(x, y):
                x = random.uniform(0, self.map_size[0])
                y = random.uniform(0, self.map_size[1])

            initial_agent_node = Node("agent_" + str(i), i, x, y)
            if self.check_cost_of_agent(initial_agent_node):
                
                self.agents.append(initial_agent_node)
            else:
                # Remove unworthy agent from the list
                self.num_agents -= 1
        print(f"# ------- Based on the distance only {self.num_agents} are useful.")

    # TODO: need to implement collision with walls for agent initial placement  
    # This function checks if a node is collision free. That means no collision with
    # any obstacle or any other initial nodes
    def is_agent_collision_free(self, x, y):

        # Checking if nodes collide with any walls
        for obstacle in self.obstacles: 
            x_min, y_min, width, height = obstacle
            
            if (x_min <= x <= x_min + width and y_min <= y <= y_min + height):
                print(f"# ------- Agent {x} {y} collides with walls")
                return False
        
        # Checking if nodes overlap - trying to have them as sparse as possible
        for other_agent in self.agents:
            cx = other_agent.x
            cy = other_agent.y
            if (x - cx) ** 2 + (x - cy) ** 2  <= self.agent_node_radius ** 2:
                print(f"# ------- Agent at {x} {y} collides with {other_agent.id}")
                return False

        return True
    
    # This function check the cost based on the distance from the agent
    # to the line between initial node and goal node 
    def check_cost_of_agent(self, agent_node):
        x0, y0 = agent_node.x, agent_node.y
        x1, y1 = self.start_node.x, self.start_node.y
        x2, y2 = self.goal_node.x, self.goal_node.y
        
        distance = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
        distance = distance / math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if distance >= self.cost_threshold:
            return False
        return True
        pass


    # This function gets creates a random node anywhere on the map
    # The idea is to try to move towards that node and explore.
    # This is part of the random exploration part
    def get_weighted_random_node(self):
        if random.random() < 0.2:
            return Node("node", self.goal_node.x, self.goal_node.y)
        else:
            x = random.uniform(0, self.map_size[0])
            y = random.uniform(0, self.map_size[1])
            if random.random() < 0.7:
                x = (x + self.goal_node.x) / 2
                y = (y + self.goal_node.y) / 2
            return Node("node", x, y)
    
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

                rand_node = self.get_weighted_random_node()
                nearest_node = self.get_nearest_node(self.trees[agent_id], rand_node)
                new_node = self.steer(nearest_node, rand_node)

                if self.is_collision_free(nearest_node, new_node):
                    near_nodes = self.find_near_nodes(self.trees[agent_id], new_node)
                    new_node = self.choose_parent(near_nodes, new_node)
                    self.trees[agent_id].append(new_node)
                    self.rewire(self.trees[agent_id], near_nodes, new_node)
                    draw_tree(self.ax, new_node, live_plot=self.live_plot)

                    if self.reached_goal(new_node, self.goal_node):
                        self.paths[agent_id] = self.generate_final_path(new_node)
                        self.goal_reached[agent_id] = True
                        draw_path(self.ax, self.paths, agent_id)

                    if self.reached_goal(new_node, self.goal_node):
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
        new_node = Node("node", from_node.x + self.step_size * math.cos(theta),
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

