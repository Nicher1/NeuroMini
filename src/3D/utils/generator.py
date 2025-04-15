import math
import random

def generate_random_polygons(map_size, num_polygons=5, z_height=10, seed=None):
    """
    Generate random convex-like polygon obstacles.
    Each polygon is defined as a list of (x, y, z) points defining both the top and bottom face.
    """
    if seed is not None:
        random.seed(seed)

    prisms = []

    for _ in range(num_polygons):
        # Random center
        cx = random.randint(20, map_size[0] - 20)
        cy = random.randint(20, map_size[1] - 20)
        # Number of vertices
        num_vertices = random.randint(3, 6)

        # Radius for size
        radius = random.randint(8, 20)

        bottom_face = []
        top_face = []

        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices + random.uniform(-0.2, 0.2)
            r = radius * random.uniform(0.7, 1.2)
            # Clip to map and define position
            x = max(0, min(cx + r * math.cos(angle), map_size[0]))
            y = max(0, min(cy + r * math.sin(angle), map_size[1]))
            bottom_face.append((x, y, 0))
            top_face.append((x, y, z_height))

        prisms.append({"bottom": bottom_face, "top": top_face})

    return prisms

def generate_random_rectangles(map_size, num_cubes=10, min_size=5, max_size=20, z_height=10):
    """
    Generate random 3D cuboid obstacles.
    Each cuboid is defined as (x, y, z, width, height, depth)
    """
    cubes = []

    for _ in range(num_cubes):
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        x = random.randint(0, map_size[0] - w)
        y = random.randint(0, map_size[1] - h)
        z = 0 # Base level for all the cuboids

        cubes.append((x, y, z, w, h, z_height))

    return cubes

# In here there are various functions that you can use to generate map to test RRT.

def generate_labyrinth_maze(map_size, wall_size=3, cell_size=20, depth=None):
    if depth is None:
        depth = map_size[2] if len(map_size) > 2 else 10

    width, height = map_size[0] // cell_size, map_size[1] // cell_size
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
                    walls_to_remove.add((wx, wy, 0, wall_size, cell_size, depth))
                elif dx == -1:
                    wx = x * cell_size
                    wy = y * cell_size
                    walls_to_remove.add((wx, wy, 0, wall_size, cell_size, depth))
                elif dy == 1:
                    wx = x * cell_size
                    wy = (y + 1) * cell_size
                    walls_to_remove.add((wx, wy, 0, cell_size, wall_size, depth))
                elif dy == -1:
                    wx = x * cell_size
                    wy = y * cell_size
                    walls_to_remove.add((wx, wy, 0, cell_size, wall_size, depth))

                visit(nx, ny)

    # Step 1: generate all walls between cells
    walls_to_remove = set()
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                walls.append((x * cell_size + cell_size, y * cell_size, 0, wall_size, cell_size, depth))  # vertical
            if y < height - 1:
                walls.append((x * cell_size, y * cell_size + cell_size, 0, cell_size, wall_size, depth))  # horizontal

    # Step 2: carve a perfect maze
    visit(0, 0)

    # Step 3: remove carved wall segments from wall list
    final_walls = [w for w in walls if w not in walls_to_remove]

    return final_walls



def generate_diagonal_wall(map_size, block_size=2, wall_thickness=5, z_height=10):
    obstacles = []
    gap_top_y = map_size[1] - 10
    gap_bottom_y = 10

    z_height = map_size[2] if len(map_size) > 2 else z_height

    for i in range(0, int(math.hypot(*map_size[:2])), block_size):
        t = i / math.hypot(*map_size[:2])
        x = t * map_size[0]
        y = map_size[1] - t * map_size[1]
        if y > gap_top_y or y < gap_bottom_y:
            continue
        rect_x = x - block_size / 2
        rect_y = y - wall_thickness / 2
        obstacles.append((rect_x, rect_y, 0, block_size, wall_thickness, z_height))

    return obstacles



def generate_map(map_type, map_size, num_obstacles=None, z_height=10):
    if map_type == "diagonal":
        return generate_diagonal_wall(map_size, z_height=z_height)
    elif map_type == "rectangles":
        return generate_random_rectangles(
            map_size,
            num_cubes=num_obstacles if num_obstacles is not None else 10,
            z_height=z_height
        )
    elif map_type == "random_polygons":
        return generate_random_polygons(
            map_size,
            num_polygons=num_obstacles if num_obstacles is not None else 5,
            z_height=z_height
        )
    elif map_type == "labyrinth":
        return generate_labyrinth_maze(map_size, depth=z_height)
    elif map_type == "empty":
        return []
    else:
        raise ValueError(f"Unknown map type: {map_type}")

