import matplotlib.pyplot as plt


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
        
        
        ax.plot(agent.initial_node.x, agent.initial_node.y, 'o', color=agent.color, label=f'Start {i+1}')
    ax.plot(goal.x, goal.y, 'ro', label=f'Goal {i+1}')
    
    if obstacle_type == "wall":
        draw_rectangle_obstacles(ax, obstacles)
    else:
        draw_circle_obstacles(ax, obstacles)

    ax.legend(loc='upper left', fontsize=12)

def draw_circle_obstacles(ax, obstacles):
    for (ox, oy, size) in obstacles:
        circle = plt.Circle((ox, oy), size, color='gray')
        ax.add_artist(circle)

def draw_rectangle_obstacles(ax, obstacles):
    for (x, y, w, h) in obstacles:
        rect = plt.Rectangle((x, y), w, h, color='gray', alpha=0.7)
        ax.add_patch(rect)

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
