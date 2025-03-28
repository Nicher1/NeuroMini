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
    def __init__(self, id, node, color):

        # Unique id for the agent
        self.id = id

        self.initial_node = node
        # List of all nodes that this agent created
        self.nodes = [self.initial_node]
        self.path = []

        self.goal_reached = False
        self.goal_node = None
        
        # With what agent connects?
        self.linked_to = None
        
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
                    step_size=1.0, max_iter=500, live_plot=False, debug=False, fig=None, ax=None, ):
        

        # Map properties
        self.map_size = map_size
        self.map_type = map_type
        self.obstacles = generate_map(map_type, map_size, num_obstacles)
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
        self.search_radius = 4.0
        self.link_radius = 10.0
        
        # Initialise agents
        self.agents = [Agent(0, self.start_node, AGENT_COLORS[0 % len(AGENT_COLORS)])]
        self.num_agents = num_agents
        
        # Agent generation variables
        self.cost_threshold = (self.map_size[0] + self.map_size[1]) / 2
        self.agent_node_radius = 10 # Helps checking collisions for agent placement

        self.generate_agents()

        # self.trees = [[self.agents[i]] for i in range(self.num_agents)]
        self.paths = [None] * self.num_agents
        self.goal_reached = [False] * self.num_agents


        # Visualization and debug setup

        self.live_plot = live_plot
        self.debug = debug

        self.fig, self.ax = plt.subplots()
        setup_visualization(self.ax, self.agents, self.goal_node, self.map_size, self.obstacle_type, self.obstacles)


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

            # Repeat until the agent is placed on the map
            while not self.is_agent_collision_free(x, y):
                x = random.uniform(0, self.map_size[0])
                y = random.uniform(0, self.map_size[1])

            initial_agent_node = Node("start", i, x, y)
            if self.check_cost_of_agent(initial_agent_node):
                color = AGENT_COLORS[i % len(AGENT_COLORS)]
                agent = Agent(i, initial_agent_node, color)
                
                self.agents.append(agent)
            else:
                # Remove unworthy agent from the list
                self.num_agents -= 1
        print(f"# ------- Based on the distance only {self.num_agents} are useful.")

   
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
        return Node(random.uniform(0, self.map_size[0]), random.uniform(0, self.map_size[1]))


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
    
    # Main Algorithm is here - from where we call functions.
    def plan(self):
        # Start the timer for each agent
        for agent in self.agents:
            agent.start_time = time.time() 

        for iter_count in range(self.max_iter):
            for agent in self.agents:
                
                if agent.goal_reached:
                    continue

                # Generate a random node arround the map - favoring goal direction
                rand_node = self.get_weighted_random_node(agent)
                # Get the nearest node to the random node
                nearest_node = self.get_nearest_node(agent.nodes, rand_node)
                # New node appears on the map - this is to be linked with the agent
                exploratory_step = self.step_size * (2 if random.random() < 0.1 else 1)
                new_node = self.steer(agent.id, nearest_node, rand_node, self.step_size)

                if self.is_collision_free(nearest_node, new_node):
                    

                    # Rewireing
                    near_nodes = self.find_near_nodes(agent.nodes, new_node)                    
                    new_node = self.choose_parent(agent.id, near_nodes, new_node)
                    agent.add_node(new_node)
                    self.rewire(agent.nodes, near_nodes, new_node)
                    
                    draw_tree(self.ax, new_node, color=agent.color, live_plot=self.live_plot)

                    # # Check if agent met another agent
                   
                    linked = self.check_and_link_agents(agent, new_node)
                    if linked:
                        if self.debug:           
                            print(f"Link Agent {agent.id}")

                        draw_path(self.ax, agent.path, color="red")
                        
                    
                 
                    if self.reached_goal(new_node, self.goal_node):
                        if self.debug:
                            print(f"Agent {agent.id} reached the goal")

                        agent.path = self.generate_final_path(new_node)
                        agent.goal_reached = True
                        agent.goal_node = new_node

                      
                        # Save result info
                        duration = time.time() - agent.start_time
                        path_len = self.compute_path_length(agent.path)

                        agent.results["iterations"] = iter_count
                        agent.results["path_length"] = path_len
                        agent.results["time"] = duration

                        if agent.results["linked"]:
                            draw_path(self.ax, agent.path, color='black', linestyle='--', label=f"Agent {agent.id} Linked Path")
                        else:
                            draw_path(self.ax, agent.path, color="blue", label=f"Agent {agent.id} Path")


        print("\n--- Agent Results ---")
        for agent in self.agents:
            print(f"Agent {agent.id} | Linked: {agent.results['linked']} | Path length: {agent.results['path_length']:.2f} | Time: {agent.results['time']}")

                
    
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
    
    def generate_final_path(self, end_node, start_node=None):
        path = []
        node = end_node
        while node and (start_node is None or node != start_node):
            path.append((node.x, node.y))
            node = node.parent
        if start_node:
            path.append((start_node.x, start_node.y))
        return path[::-1]
    
    def find_near_nodes(self, tree, new_node, radius=10.0):
        near_nodes = []
        for node in tree:
            dist = math.hypot(node.x - new_node.x, node.y - new_node.y)
            if dist <= radius:
                near_nodes.append(node)
        return near_nodes
    
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
        return new_node
    
    def distance(self, node1, node2):
        return math.hypot(node1.x - node2.x, node1.y - node2.y)

    def rewire(self, tree, near_nodes, new_node):
        for node in near_nodes:
            new_cost = new_node.cost + self.distance(new_node, node)
            if new_cost < node.cost and self.is_collision_free(new_node, node):
                node.parent = new_node
                node.cost = new_cost


    # CMN Functions From there
    def check_and_link_agents(self, agent_a, new_node):
        for agent_b in self.agents:
            if agent_b.id == agent_a.id:
                continue  # skip self

            for node_b in agent_b.nodes:
                if self.distance(new_node, node_b) <= self.link_radius:
                    
                    if not self.is_collision_free(new_node, node_b):
                       
                        continue
                    
                    if not agent_b.goal_reached:
                        continue  # we only link to agents that reached the goal

                    # Create linked path: A to meeting point, then B to goal
                    path_to_meeting = self.generate_final_path(new_node)
                    path_from_meeting = self.generate_final_path(agent_b.goal_node, start_node=node_b)

                    full_linked_path = path_to_meeting + path_from_meeting

                    linked_path_length = self.compute_path_length(full_linked_path)

                    # If agent A has no goal path or this is better
                    if not agent_a.goal_reached or linked_path_length < agent_a.results["path_length"]:
                        # Accept the link
                        agent_a.goal_reached = True
                        agent_a.linked_to = agent_b.id
                        agent_a.path = full_linked_path
                        agent_a.results["path_length"] = linked_path_length
                        agent_a.results["linked"] = True
                        return True
        return False
    
    def generate_linked_path(self, agent_a, meeting_node_a, meeting_node_b, agent_b):
        # Path from agent A's tree to meeting point
        path_to_meet = self.generate_final_path(meeting_node_a)

        # Path from agent B's meeting node to its goal
        path_from_meet = []
        node = meeting_node_b
        while node:
            path_from_meet.append((node.x, node.y))
            node = node.parent
        path_from_meet.reverse()

        # Combine the paths
        return path_to_meet + path_from_meet

    def animate(self):
        plt.show()

    def export_plot_as_base64(self):
        buffer = io.BytesIO()
        self.fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        return img_str
    
# Execution

