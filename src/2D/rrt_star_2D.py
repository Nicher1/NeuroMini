#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.spatial
import math
import random
import argparse
import time

# Node class representing a state in the space
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0

# Multi-Agent RRT* (CMN-RRT*)
class CMN_RRTStar:
    def __init__(self, start_positions, goal_positions, num_agents, num_obstacles, map_size, map_type="empty", step_size=1.0, max_iter=500, live_plot=False):
        
        # Map properties
        self.map_size = map_size
        self.map_type = map_type
        self.obstacles = self.generate_map()
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
        self.setup_visualization()

    def generate_diagonal_wall(self):
        wall_thickness = 7  # how wide the wall is (perpendicular to the diagonal)
        block_size = 2      # how long each rectangle is along the diagonal
        obstacles = []

        gap_top_y = self.map_size[1] - 10
        gap_bottom_y = 10

        for i in range(0, int(math.hypot(*self.map_size)), block_size):
            # Calculate point along diagonal from (0, height) to (width, 0)
            t = i / math.hypot(*self.map_size)
            x = t * self.map_size[0]
            y = self.map_size[1] - t * self.map_size[1]

            # Skip if inside top or bottom gap
            if y > gap_top_y or y < gap_bottom_y:
                continue

            # Rectangle axis-aligned centered at (x, y)
            rect_x = x - block_size / 2
            rect_y = y - wall_thickness / 2
            obstacles.append((rect_x, rect_y, block_size, wall_thickness))

        return obstacles
    
    def generate_labyrinth_maze(self, wall_size=3, cell_size=20):
        width, height = self.map_size[0] // cell_size, self.map_size[1] // cell_size
        maze = [[False] * width for _ in range(height)]  # visited cells
        walls = []

        def visit(x, y):
            maze[y][x] = True
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not maze[ny][nx]:
                    # Remove wall between (x,y) and (nx,ny)
                    if dx == 1:
                        wx = (x + 1) * cell_size
                        wy = y * cell_size
                        walls_to_remove.add((wx, wy, wall_size, cell_size))
                    elif dx == -1:
                        wx = x * cell_size
                        wy = y * cell_size
                        walls_to_remove.add((wx, wy, wall_size, cell_size))
                    elif dy == 1:
                        wx = x * cell_size
                        wy = (y + 1) * cell_size
                        walls_to_remove.add((wx, wy, cell_size, wall_size))
                    elif dy == -1:
                        wx = x * cell_size
                        wy = y * cell_size
                        walls_to_remove.add((wx, wy, cell_size, wall_size))

                    visit(nx, ny)

        # Step 1: generate all walls between cells
        walls_to_remove = set()
        for y in range(height):
            for x in range(width):
                if x < width - 1:
                    walls.append((x * cell_size + cell_size, y * cell_size, wall_size, cell_size))  # vertical
                if y < height - 1:
                    walls.append((x * cell_size, y * cell_size + cell_size, cell_size, wall_size))  # horizontal

        # Step 2: carve a perfect maze
        visit(0, 0)

        # Step 3: remove carved wall segments from wall list
        final_walls = [w for w in walls if w not in walls_to_remove]

        return final_walls

    
    def generate_map(self):
        if self.map_type == "labyrinth":
            return self.generate_labyrinth_maze()
        elif self.map_type == "diagonal":
            return self.generate_diagonal_wall()
        elif self.map_type == "empty":
            return []
            

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
    
    def get_nearest_node(self, tree, rand_node):
        points = np.array([[node.x, node.y] for node in tree])
        tree_kdtree = scipy.spatial.KDTree(points)
        _, index = tree_kdtree.query([rand_node.x, rand_node.y])
        return tree[index]
    
    def setup_visualization(self):
        self.ax.set_xlim(0, self.map_size[0])
        self.ax.set_ylim(0, self.map_size[1])
        self.ax.grid(True)

        for i in range(self.num_agents):
            self.ax.plot(self.agents[i].x, self.agents[i].y, 'bo', label=f'Start {i+1}')
            self.ax.plot(self.goals[i].x, self.goals[i].y, 'ro', label=f'Goal {i+1}')
        
        if self.obstacle_type == "wall":
            self.draw_rectangle_obstacles()
        else:
            self.draw_circle_obstacles()

    def draw_circle_obstacles(self):
        for (ox, oy, size) in self.obstacles:
            circle = plt.Circle((ox, oy), size, color='gray')
            self.ax.add_artist(circle)

    def draw_rectangle_obstacles(self):
        for (x, y, w, h) in self.obstacles:
            rect = plt.Rectangle((x, y), w, h, color='gray', alpha=0.7)
            self.ax.add_patch(rect)

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
                    self.trees[agent_id].append(new_node)
                    new_node.parent = nearest_node
                    self.draw_tree(new_node)

                    if self.reached_goal(new_node, self.goals[agent_id]):
                        self.paths[agent_id] = self.generate_final_path(new_node)
                        self.goal_reached[agent_id] = True

                        # Save result info
                        duration = time.time() - start_time[agent_id]
                        path_len = self.compute_path_length(self.paths[agent_id])

                        self.agents_results[agent_id]["iterations"] = iter_count
                        self.agents_results[agent_id]["path_length"] = path_len
                        self.agents_results[agent_id]["time"] = duration

                        self.draw_path(agent_id)
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
    
    def draw_tree(self, node):
        if node.parent:
            self.ax.plot([node.x, node.parent.x], [node.y, node.parent.y], "-b")
            if self.live_plot:
                plt.pause(0.01)
    
    def draw_path(self, agent_id):
        if self.paths[agent_id]:
            self.ax.plot([x[0] for x in self.paths[agent_id]], [x[1] for x in self.paths[agent_id]], '-g', label=f'Path {agent_id+1}')
    
    def animate(self):
        plt.show()
    
# Execution

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CMN-RRT* path planner (2D)")

    parser.add_argument("--num_agents", type=int, default=1,
                        help="Number of agents to plan paths for")
    parser.add_argument("--num_obstacles", type=int, default=15,
                        help="Number of wall segments (ignored if using fixed wall layout)")
    parser.add_argument("--map_size", nargs=2, type=int, default=[200, 200],
                        help="Map size as width height")
    parser.add_argument("--step_size", type=float, default=1.0,
                        help="Step size for each tree expansion")
    parser.add_argument("--live_plot", action="store_true",
                    help="Enable live tree drawing during planning")
    parser.add_argument("--map_type", choices=["empty", "diagonal", "labyrinth"], default="empty",
                    help="Choose the type of map layout")

    args = parser.parse_args()

    # Auto-generate start/goal positions (evenly spaced)
    start_positions = [[5 + i * 5, 5] for i in range(args.num_agents)]
    goal_positions = [[args.map_size[0] - 10, args.map_size[1] - 10] for i in range(args.num_agents)]

    cmn_rrt_star = CMN_RRTStar(
        start_positions,
        goal_positions,
        args.num_agents,
        args.num_obstacles,
        args.map_size,
        step_size=args.step_size,
        max_iter=10000,
        live_plot=args.live_plot,
        map_type=args.map_type
    )


    cmn_rrt_star.plan()
    cmn_rrt_star.animate()