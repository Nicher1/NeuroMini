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
            plt.pause(0.01)

def draw_path(ax, path, color='green', linewidth=3, linestyle='-', label=None):
    if path:
        ax.plot(
            [x[0] for x in path],
            [x[1] for x in path],
            linestyle,
            color=color,
            linewidth=linewidth,
            label=label
        )
