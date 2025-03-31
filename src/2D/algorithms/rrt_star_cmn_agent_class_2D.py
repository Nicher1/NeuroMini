import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.spatial
import math
import random
import time
import io
import base64

from utils.generator import generate_map
from utils.plotter import *
        
class Node:
    def __init__(self, label, agent_id, x, y):
        self.label = label
        self.agent_id = agent_id # 0, 1, 2 ... -1 means no agent (goal node)
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0
        self.sons = []

class Agent:
    def __init__(self, id, node, color):

        # Unique id for the agent
        self.id = id

        self.initial_node = node
        # List of all nodes that this agent created
        self.nodes = [self.initial_node]
        self.path = []

        self.goal_reached = False
        
        # CMN Variables
        self.linked_from = None   # Parent agent (the one that links to this agent)
        self.linked_to = []       # List of agents that linked from this agent
        
        # Results per agent
        self.start_time = 0.0
        self.results = {
            "iterations": 0,
            "path_length": float('inf'),
            "time": 0.0,
            "linked": False
        }

        # Plotting variables
        self.color = color

    def add_node(self, node):
        self.nodes.append(node)

# Multi-Agent RRT* (CMN-RRT*)
class CMNRRTStarV2:
    def __init__(self, start_position, goal_position, num_obstacles,
                  num_agents, map_size, map_type="empty",
                    step_size=1.0, max_iter=500, live_plot=False, debug=False, fig=None, ax=None, map_obstacles=None):
        self.debug = debug

        # Map properties
        self.map_size = map_size
        self.map_type = map_type
        if map_obstacles == None:
            self.obstacles = generate_map(map_type, map_size, num_obstacles)
        else:
            self.obstacles = map_obstacles
        if map_type == "random_polygons":
            self.obstacle_type = "polygon"
        else:
            self.obstacle_type = "wall"

        self.goal_region_radius = 4

        self.start_node = Node("start", 0, start_position[0], start_position[1])
        self.goal_node = Node("goal", -1, goal_position[0], goal_position[1])
  
        # Algorithm variables
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = step_size
        self.link_radius = 4.0
        
        
        # Initialise agents
        self.agents = [Agent(0, self.start_node, AGENT_COLORS[0 % len(AGENT_COLORS)])]
        self.num_agents = num_agents
        
        # Agent generation variables
        self.cost_threshold =  10 # (self.map_size[0] + self.map_size[1]) / 10
        self.agent_node_radius = 20 # Helps checking collisions for agent placement

        self.generate_agents()

        # self.trees = [[self.agents[i]] for i in range(self.num_agents)]
        self.paths = [None] * self.num_agents
        self.goal_reached = [False] * self.num_agents


        # Visualization and debug setup

        self.live_plot = live_plot
        
        self.fig, self.ax = plt.subplots()
        setup_visualization(self.ax, self.agents, self.goal_node, self.map_size, self.obstacle_type, self.obstacles)

        # Planning results
        self.total_planning_time = 0
        self.total_path_length = float('inf')
        self.contributing_agents = []


        self.goal_merge_threshold = 30  # or tune based on map size
        



    # This function generates self.num_agents. These agents are actually
    # initial nodes for CMN_RRT. If the node can't be placed due to collision
    # the function will try to find another random position
    # A distance is calculated between the node and the starting node and goal node
    # and based on this the node is gonna be used or not
    def generate_agents(self):

        # Starts from 1 because Agent 0 is created in the starting node
        for i in range(1, self.num_agents):
            x = random.uniform(0, self.map_size[0])
            y = random.uniform(0, self.map_size[1])
            initial_agent_node = Node("start", i, x, y)

            count = 0
            # Repeat until the agent is placed on the map
            while not self.is_agent_collision_free(x, y) or not self.check_cost_of_agent(initial_agent_node):
                x = random.uniform(0, self.map_size[0])
                y = random.uniform(0, self.map_size[1])
                initial_agent_node = Node("start", i, x, y)
                if count >= 500:
                    break
                count += 1
                    
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            agent = Agent(i, initial_agent_node, color)
            
            self.agents.append(agent)    
   
   
    # This function checks if a node is collision free. That means no collision with
    # any obstacle or any other initial nodes
    def is_agent_collision_free(self, x, y):

        # Checking if nodes collide with any walls
        if self.obstacle_type == "wall":
            for obstacle in self.obstacles:
                x_min, y_min, width, height = obstacle
                if x_min <= x <= x_min + width and y_min <= y <= y_min + height:
                    return False

        elif self.obstacle_type == "polygon":
            from matplotlib.path import Path
            for polygon in self.obstacles:
                path = Path(polygon)
                if path.contains_point((x, y)):
                    return False
        
        # Checking if nodes overlap - trying to have them as sparse as possible
        for other_agent in self.agents:
            cx = other_agent.initial_node.x
            cy = other_agent.initial_node.y
            if (x - cx) ** 2 + (x - cy) ** 2  <= self.agent_node_radius ** 2:

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

        if distance > self.cost_threshold:
            return False
        return True
        

    # --- START ---- Exploration Functions ---- 
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
    
    def rewire(self, tree, near_nodes, new_node):
        for node in near_nodes:
            new_cost = new_node.cost + self.distance(new_node, node)
            if new_cost < node.cost and self.is_collision_free(new_node, node):
                node.parent = new_node
                node.cost = new_cost

    def distance(self, node1, node2):
        return math.hypot(node1.x - node2.x, node1.y - node2.y)
    
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
    
    # --- END ---- Exploration Functions ---- 

    # --- START ---- Tree generation functions ----

    # This function get the nearest node to the random node
    # previously generated. 
    def get_nearest_node(self, tree, rand_node):
        points = np.array([[node.x, node.y] for node in tree])
        tree_kdtree = scipy.spatial.KDTree(points)
        _, index = tree_kdtree.query([rand_node.x, rand_node.y])
        return tree[index]
    
    def steer(self, agent_id, from_node, to_node, step_size):
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_node = Node("node", agent_id, from_node.x + step_size * math.cos(theta),
                        from_node.y + self.step_size * math.sin(theta))
        new_node.parent = from_node
        return new_node
    
    def choose_parent(self, agent_id, near_nodes, new_node):
        best_parent = None
        best_cost = float('inf')

        for node in near_nodes:
            temp_node = self.steer(agent_id, node, new_node, self.step_size)
            if self.is_collision_free(node, temp_node):
                cost = node.cost + self.distance(node, temp_node)
                if cost < best_cost:
                    best_parent = node
                    best_cost = cost

        if best_parent:
            new_node.parent = best_parent
            new_node.cost = best_cost
            best_parent.sons.append(new_node)  # Add the new node to the parent's sons list
        return new_node

    # --- START ---- CMN algorithm and everything it needs ----

    # Main Algorithm is here - from where we call functions.
    def plan(self):
        # Start the timer that ends when the starting node is connected to the goal
        start_time = time.time()

        # For each step
        for iter_count in range(self.max_iter):
        
            # If the first agent has arrived, finish the algorithm
            if self.agents[0].goal_reached:
                # Stop the timer
                if self.agents[0].results["time"] == 0.0:
                    self.agents[0].results["time"] = time.time() - start_time

                # Generate the final path and calculate its cost
                self.agents[0].path, self.agents[0].results["path_length"] = self.generate_final_path(
                    self.goal_node, self.start_node
                )
                self.agents[0].results["iterations"] = iter_count

                # Convert the path to a list of (x, y) tuples
                path_coordinates = [(node.x, node.y) for node in self.agents[0].path]

            

               
                draw_path(self.ax, path_coordinates, color='red', linewidth=3, linestyle='--', label="Final Path")
                plt.legend()
                plt.pause(0.01)  # Update the plot

      
             
                print(f"Final Path Found!")
                print(f"Time Taken: {self.agents[0].results['time']:.2f} seconds")
                print(f"Path Length: {self.agents[0].results['path_length']:.2f}")
                print(f"Iterations: {self.agents[0].results['iterations']}")
                break

            # For each agent
            for agent in self.agents:
                if not agent.goal_reached:
                    # Generate a random node around the map - favoring goal direction
                    rand_node = self.get_weighted_random_node(agent)
                    nearest_node = self.get_nearest_node(agent.nodes, rand_node)
                    new_node = self.steer(agent.id, nearest_node, rand_node, self.step_size)

                    if self.is_collision_free(nearest_node, new_node):
                        # Rewiring
                        near_nodes = self.find_near_nodes(agent.nodes, new_node)
                        new_node = self.choose_parent(agent.id, near_nodes, new_node)
                        agent.add_node(new_node)
                        self.rewire(agent.nodes, near_nodes, new_node)

                        draw_tree(self.ax, new_node, color=agent.color, live_plot=self.live_plot)
                        self.link_near_nodes(agent, new_node)

                        # Check if the agent reached the goal
                        if self.reached_goal(new_node, self.goal_node):
                            agent.goal_reached = True
                    
                            if agent.id != 0:
                                self.goal_node = new_node

        # If no path is found after max iterations
        if self.debug:
            if not self.agents[0].goal_reached:
                print("No path found within the maximum iterations.")

    def link_near_nodes(self, agent, new_node):
        # Check if the new node is close to any other node from other agents
        for other_agent in self.agents:
            if other_agent.id != agent.id:
                for other_node in other_agent.nodes:
                    if self.distance(new_node, other_node) < self.link_radius and self.is_collision_free(new_node, other_node):
                        new_node.sons.append(other_node)
                        other_node.sons.append(new_node)
                        if other_agent.goal_reached:
                            agent.goal_reached = True
                        break

    def reached_goal(self, node, goal):
        return np.linalg.norm([node.x - goal.x, node.y - goal.y]) < self.goal_region_radius

    def generate_final_path(self, end_node, start_node=None):
        # Initialize distances and previous nodes
        distances = {end_node: 0}
        previous_nodes = {end_node: None}
        unvisited = [end_node]

        # Dijkstra's algorithm
        while unvisited:

            # Get the node with the smallest distance
            current_node = min(unvisited, key=lambda node: distances[node])
            unvisited.remove(current_node)

            # If we reached the start node, stop
            if start_node and current_node == start_node:
                break

            # Traverse sons and parent
            neighbors = current_node.sons[:]
            if current_node.parent:
                neighbors.append(current_node.parent)

            for neighbor in neighbors:
                cost = self.distance(current_node, neighbor)
                new_distance = distances[current_node] + cost

                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
                    unvisited.append(neighbor)

        # Reconstruct the path
        path = []
        current_node = start_node if start_node else self.start_node
        while current_node:
            path.append(current_node)
            current_node = previous_nodes.get(current_node)

        path.reverse()
        return path, distances.get(start_node, float('inf'))


    def find_near_nodes(self, tree, new_node, radius=10.0):
        near_nodes = []
        for node in tree:
            dist = math.hypot(node.x - new_node.x, node.y - new_node.y)
            if dist <= radius:
                near_nodes.append(node)
        return near_nodes

    def animate(self):
        plt.show()

    def export_plot_as_base64(self):
        buffer = io.BytesIO()
        self.fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        return img_str
    
# Execution

