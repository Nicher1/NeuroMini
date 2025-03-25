import math
import random

# In here there are various functions that you can use to generate map to test RRT.
def generate_labyrinth_maze(map_size, wall_size=3, cell_size=20):
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


def generate_diagonal_wall(map_size, block_size=2, wall_thickness=5):
    obstacles = []
    gap_top_y = map_size[1] - 10
    gap_bottom_y = 10

    for i in range(0, int(math.hypot(*map_size)), block_size):
        t = i / math.hypot(*map_size)
        x = t * map_size[0]
        y = map_size[1] - t * map_size[1]
        if y > gap_top_y or y < gap_bottom_y:
            continue
        rect_x = x - block_size / 2
        rect_y = y - wall_thickness / 2
        obstacles.append((rect_x, rect_y, block_size, wall_thickness))

    return obstacles

def generate_map(map_type, map_size, num_obstacles=None):
    if map_type == "diagonal":
        return generate_diagonal_wall(map_size)
    elif map_type == "labyrinth":
        return generate_labyrinth_maze(map_size)
    elif map_type == "empty":
        return []
    else:
        raise ValueError(f"Unknown map type: {map_type}")