import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon as MplPolygon
from mpl_toolkits.mplot3d import Axes3D  # Needed to enable 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

AGENT_COLORS = [
    'orange', 'green', 'purple', 'cyan',
    'pink', 'lime', 'magenta', 'brown', 'teal', 'navy',
    'gold', 'olive', 'salmon', 'turquoise', 'gray', 'black'
]

def setup_visualization(ax, agents, goal, map_size, obstacle_type, obstacles):
    ax.set_xlim(0, map_size[0])
    ax.set_ylim(0, map_size[1])
    ax.set_zlim(0, map_size[2] if len(map_size) > 2 else 20)  # set Z limit

    for _, agent in enumerate(agents):
        ax.scatter(agent.initial_node.x, agent.initial_node.y, agent.initial_node.z, color=agent.color, label=f"Agent_{agent.id}" if agent.id != 0 else "Start")
    ax.scatter(goal.x, goal.y, goal.z, color='red', label='Goal')

    if obstacle_type == "wall":
        draw_rectangle_obstacles(ax, obstacles)
    elif obstacle_type == "olygon":
        draw_polygon_obstacles(ax, obstacles)
    
    ax.legend(loc='lower right', fontsize=10)

def draw_circle_obstacles(ax, obstacles):
    for (ox, oy, size) in obstacles:
        circle = Circle((ox, oy), size, color='gray')
        ax.add_artist(circle)



def draw_rectangle_obstacles(ax, obstacles):
    for obstacle in obstacles:
        if len(obstacle) == 6:
            x, y, z, w, h, d = obstacle
        elif len(obstacle) == 5:
            x, y, w, h, z = obstacle
            d = 5
        elif len(obstacle) == 4:
            x, y, w, h = obstacle
            z, d = 0, 5
        else:
            raise ValueError(f"Unexpected obstacle format: {obstacle}")
        draw_cuboid(ax, x, y, z, w, h, d)


def draw_cuboid(ax, x, y, z, w, h, d, color='gray', alpha=0.5):
    corners = [
        [x, y, z],
        [x+w, y, z],
        [x+w, y+h, z],
        [x, y+h, z],
        [x, y, z+d],
        [x+w, y, z+d],
        [x+w, y+h, z+d],
        [x, y+h, z+d],
    ]
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],
        [corners[4], corners[5], corners[6], corners[7]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[2], corners[3], corners[7], corners[6]],
        [corners[1], corners[2], corners[6], corners[5]],
        [corners[4], corners[7], corners[3], corners[0]],
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='k', alpha=alpha))





def draw_polygon_obstacles(ax, prisms, color='gray'):
    for prism in prisms:
        verts = []

        # Add bottom face
        bottom = [(x, y, z) for (x, y, z) in prism["bottom"]]
        verts.append(bottom)

        # Add top face
        top = [(x, y, z) for (x, y, z) in prism["top"]]
        verts.append(top)

        # Add vertical sides
        for i in range(len(bottom)):
            j = (i + 1) % len(bottom)
            side = [
                bottom[i],
                bottom[j],
                top[j],
                top[i]
            ]
            verts.append(side)

        collection = Poly3DCollection(verts, alpha=0.3, facecolor=color)
        ax.add_collection3d(collection)

def draw_tree(ax, node, color='b', live_plot=False, label=None):
    if node.parent:
        ax.plot(
            [node.x, node.parent.x],
            [node.y, node.parent.y],
            [node.z, node.parent.z],
            color=color,
            linewidth=1,
            label = label
        )
        if live_plot:
            plt.pause(0.001)

def draw_path(ax, path, color='green', linewidth=3, linestyle='-', label=None, live_plot=True):
    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]
        ax.plot(
            xs, ys, zs,
            linestyle = linestyle,
            color=color,
            linewidth=linewidth,
            label=label
        )
    if live_plot:
        plt.pause(0.01)


def save_path_plot(planner, algorithm_name, run_index, output_dir="results/", multi_agent=False):
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

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
        draw_path(ax, planner.agents[0].path, linestyle='--', color="red", label="Agent 0")

    if multi_agent:
        path_coordinates = [(node.x, node.y) for node in planner.agents[0].path]
        draw_path(ax, path_coordinates, color='red', linewidth=3, linestyle='--', label="Final Path")

    ax.set_title(f"{algorithm_name.upper()} - Run {run_index + 1}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{algorithm_name}_run{run_index + 1}.png")
plt.close()
