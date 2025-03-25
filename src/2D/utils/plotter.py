



import matplotlib.pyplot as plt

def setup_visualization(ax, agents, goals, map_size, obstacle_type, obstacles):
    ax.set_xlim(0, map_size[0])
    ax.set_ylim(0, map_size[1])
    ax.grid(True)

    for i, agent in enumerate(agents):
        ax.plot(agent.x, agent.y, 'bo', label=f'Start {i+1}')
        ax.plot(goals[i].x, goals[i].y, 'ro', label=f'Goal {i+1}')
    
    if obstacle_type == "wall":
        draw_rectangle_obstacles(ax, obstacles)
    else:
        draw_circle_obstacles(ax, obstacles)

def draw_circle_obstacles(ax, obstacles):
    for (ox, oy, size) in obstacles:
        circle = plt.Circle((ox, oy), size, color='gray')
        ax.add_artist(circle)

def draw_rectangle_obstacles(ax, obstacles):
    for (x, y, w, h) in obstacles:
        rect = plt.Rectangle((x, y), w, h, color='gray', alpha=0.7)
        ax.add_patch(rect)

def draw_tree(ax, node, live_plot=False):
    if node.parent:
        ax.plot([node.x, node.parent.x], [node.y, node.parent.y], "-b")
        if live_plot:
            plt.pause(0.01)

def draw_path(ax, paths, agent_id):
    if paths[agent_id]:
        ax.plot([x[0] for x in paths[agent_id]], [x[1] for x in paths[agent_id]], '-g', label=f'Path {agent_id+1}')
