
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
    def __init__(self, label, agent_id, x, y):
        self.label = label
        self.agent_id = agent_id # 0, 1, 2 ... -1 means no agent (goal node)
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0

class Agent:
    def __init__(self, id, node, color='blue'):
        self.id = id
        self.initial_node = node
        self.nodes = [node]
        self.goal_reached = False
        self.goal_node = None
        self.path = []
        self.color = color
        self.start_time = 0.0
        self.results = {
            "iterations": 0,
            "path_length": float('inf'),
            "time": 0.0,
        }

    def add_node(self, node):
        self.nodes.append(node)

class RRT:
    def __init__(self, start_position, goal_position, num_agents, num_obstacles, map_size, map_type="empty", step_size=1.0, max_iter=500, live_plot=False, pregenerate_map=None):
        
        # Map properties
        self.map_size = map_size
        self.map_type = map_type
        
        if pregenerate_map == None:
            self.obstacles = generate_map(map_type, map_size, num_obstacles)
        else:
            self.obstacles = pregenerate_map

        if map_type == "random_polygons":
            self.obstacle_type = "polygon"
        else:
            self.obstacle_type = "wall"

        self.goal_region_radius = 4

        self.start_node = Node("start", 0, start_position[0], start_position[1])
        self.goal_node = Node("goal", -1, goal_position[0], goal_position[1])
        
        # Algorithm properties
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = 4.0
        
        # Initialise agents
        self.agents = [Agent(0, self.start_node, AGENT_COLORS[0 % len(AGENT_COLORS)])]
        # for i in range(num_agents):
        #     node = Node(start_position[0], start_position[1])
        #     color = AGENT_COLORS[i % len(AGENT_COLORS)] if 'AGENT_COLORS' in globals() else 'blue'
        #     agent = Agent(i, node, color=color)
        #     self.agents.append(agent)

        # Visualization setup
        self.live_plot = live_plot
        self.fig, self.ax = plt.subplots()
        setup_visualization(self.ax, self.agents, self.goal_node, self.map_size, self.obstacle_type, self.obstacles)

        self.total_planning_time = 0
            

    # --- START ---- Exploration Functions ---- 
    # Get the distance between two nodes.
    def distance(self, node1, node2):
        return math.hypot(node1.x - node2.x, node1.y - node2.y)


    def dynamic_goal_bias(self, agent):
        if len(agent.nodes) < 10:
            return 0.01  # explore early
        elif agent.goal_reached:
            return 0.0
        else:
            return 0.1  # gradually increase    

    def get_exploration_node(self, agent):
        for _ in range(30):
            x = random.uniform(0, self.map_size[0])
            y = random.uniform(0, self.map_size[1])
            rand_node = Node("explore", agent.id, x, y)
   
            # Ensure sample is far from current tree 
            nearest = self.get_nearest_node(agent.nodes, rand_node)
            dist = self.distance(nearest, rand_node)
            if dist > self.step_size * 1.5:
                return rand_node

        # fallback
        return Node("explore", agent.id, random.uniform(0, self.map_size[0]), random.uniform(0, self.map_size[1]))


    # This function gets creates a random node anywhere on the map
    # The idea is to try to move towards that node and explore.
    # This is part of the random exploration part
    def get_weighted_random_node(self, agent):
        # Dynamic goal bias increases as agent's tree grows
        if agent.goal_reached:
            return self.goal_node

        goal_bias = self.dynamic_goal_bias(agent)
        if random.random() < goal_bias:
            return self.goal_node

        # Exploration-aware sample
        return self.get_exploration_node(agent)
    
    # --- END ---- Exploration Functions ---- 
    
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
        for agent in self.agents:
            agent.start_time = time.time() 

        for iter_count in range(self.max_iter):
            for agent in self.agents:
                if agent.goal_reached:
                    continue

                rand_node = self.get_weighted_random_node(agent)
                nearest_node = self.get_nearest_node(agent.nodes, rand_node)
                new_node = self.steer(agent.id, nearest_node, rand_node, self.step_size)

                if self.is_collision_free(nearest_node, new_node):
          
                    agent.add_node(new_node)
                    new_node.parent = nearest_node
                    
                    draw_tree(self.ax, new_node, color=agent.color, live_plot=self.live_plot)

                    if self.reached_goal(new_node, self.goal_node):
                        agent.path = self.generate_final_path(new_node)
                        agent.goal_reached = True

                        # Save result info
                        duration = time.time() - agent.start_time
                        path_len = self.compute_path_length(agent.path)

                        agent.results["iterations"] = iter_count
                        agent.results["path_length"] = path_len
                        agent.results["time"] = duration
                        self.total_planning_time = duration

      
        print(f"Final Path Found!")
        print(f"Time Taken: {self.agents[0].results['time']:.2f} seconds")
        print(f"Path Cost: {self.agents[0].results['path_length']:.2f}")
        print(f"Iterations: {self.agents[0].results['iterations']}")
    
        draw_path(self.ax, agent.path, color='red', linestyle='--', label=f"Agent {agent.id} Linked Path", live_plot=True) 
       
    
    def steer(self, agent_id, from_node, to_node, step_size):
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_node = Node("node", agent_id, from_node.x + step_size * math.cos(theta),
                        from_node.y + self.step_size * math.sin(theta))
        new_node.parent = from_node
        return new_node
    
    def ccw(self, A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def segments_intersect(self, A, B, C, D):
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)
    
    def segment_intersects_polygon(self, p1, p2, path, num_samples=10):
        for i in range(num_samples + 1):
            t = i / num_samples
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if path.contains_point((x, y)):
                return True
        return False
    def is_collision_free(self, from_node, to_node):
        p1 = (from_node.x, from_node.y)
        p2 = (to_node.x, to_node.y)

        if self.obstacle_type == "wall":
            for (x_min, y_min, width, height) in self.obstacles:
                x_max = x_min + width
                y_max = y_min + height

                wall_edges = [
                    ((x_min, y_min), (x_max, y_min)),  # bottom
                    ((x_max, y_min), (x_max, y_max)),  # right
                    ((x_max, y_max), (x_min, y_max)),  # top
                    ((x_min, y_max), (x_min, y_min)),  # left
                ]

                for (w1, w2) in wall_edges:
                    if self.segments_intersect(p1, p2, w1, w2):
                        return False  # collision

        elif self.obstacle_type == "polygon":
            from matplotlib.path import Path
            for polygon in self.obstacles:
                path = Path(polygon)
                # Sample the segment with intermediate points to check
                if self.segment_intersects_polygon(p1, p2, path):
                    return False

        return True
    
    def reached_goal(self, node, goal):
        return np.linalg.norm([node.x - goal.x, node.y - goal.y]) < self.goal_region_radius
    
    def generate_final_path(self, goal_node):
        path = []
        node = goal_node
        while node is not None:
            path.append([node.x, node.y])
            node = node.parent
        return path[::-1]
    

    def animate(self):
        plt.show()
  