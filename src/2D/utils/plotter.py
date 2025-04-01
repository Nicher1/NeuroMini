import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

AGENT_COLORS = [
    'orange', 'green', 'purple', 'cyan',
    'pink', 'lime', 'magenta', 'brown', 'teal', 'navy',
    'gold', 'olive', 'salmon', 'turquoise', 'gray', 'black'
]



def setup_visualization(ax, agents, goal, map_size, obstacle_type, obstacles):
    ax.set_xlim(0, map_size[0])
    ax.set_ylim(0, map_size[1])
    ax.grid(True)

    
    for i, agent in enumerate(agents):
        
        if agent.id == 0:
            ax.plot(agent.initial_node.x, agent.initial_node.y, 'o', color=agent.color, label=f"Start")
        else: 
            ax.plot(agent.initial_node.x, agent.initial_node.y, 'o', color=agent.color, label=f"Agent_{agent.id}")
    ax.plot(goal.x, goal.y, 'ro', label=f'Goal')
    
    if obstacle_type == "wall":
        draw_rectangle_obstacles(ax, obstacles)
    elif obstacle_type == "polygon":
         draw_polygon_obstacles(ax, obstacles)
    else:
        draw_circle_obstacles(ax, obstacles)

    ax.legend(
        loc='lower right', fontsize=10
    )

def draw_circle_obstacles(ax, obstacles):
    for (ox, oy, size) in obstacles:
        circle = plt.Circle((ox, oy), size, color='gray')
        ax.add_artist(circle)

def draw_rectangle_obstacles(ax, obstacles):
    for (x, y, w, h) in obstacles:
        rect = plt.Rectangle((x, y), w, h, color='gray', alpha=0.7)
        ax.add_patch(rect)

def draw_polygon_obstacles(ax, polygons, color='gray'):
    for poly in polygons:
        patch = MplPolygon(poly, closed=True, color=color, alpha=0.6)
        ax.add_patch(patch)

def draw_tree(ax, node, color='b', live_plot=False, label=None):
    if node.parent:
        ax.plot(
            [node.x, node.parent.x],
            [node.y, node.parent.y],
            color=color,
            linewidth=2,
            label=label
        )


        if live_plot:
            plt.pause(0.001)

def draw_path(ax, path, color='green', linewidth=3, linestyle='-', label=None, live_plot=True):
    if path:
        ax.plot(
            [x[0] for x in path],
            [x[1] for x in path],
            linestyle,
            color=color,
            linewidth=linewidth,
            label=label
        )
    if live_plot:
        plt.pause(0.01)


def save_path_plot(planner, algorithm_name, run_index, output_dir="results/", multi_agent=False):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots()

    setup_visualization(
        ax,
        planner.agents,
        planner.goal_node,
        planner.map_size,
        planner.obstacle_type,
        planner.obstacles
    )
                        

    for agent in planner.agents:
        
        for node in agent.nodes:
            draw_tree(ax, node, color=agent.color)
    
    if planner.agents[0].path:
        draw_path(ax, agent.path, linestyle='--', color="red", label=f"Agent {agent.id}")

    if multi_agent:
        path_coordinates = [(node.x, node.y) for node in planner.agents[0].path]
        draw_path(ax, path_coordinates, color='red', linewidth=3, linestyle='--', label="Final Path")

    ax.set_title(f"{algorithm_name.upper()} - Run {run_index + 1}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{algorithm_name}_run{run_index + 1}.png")
    plt.close()
