# NeuroMini

### Dependencies

To run this code, install the following Python packages:

```bash
pip install numpy matplotlib tqdm scipy
```

### Path-Planning Algorithms

This repository provides three path-planning algorithms:

- **RRT (Rapidly-exploring Random Trees)**: Rapidly generates a feasible path by incrementally building a tree from random points towards the goal.
    
- **RRT* (Optimised Rapidly-exploring Random Trees)**: Improves upon basic RRT by integrating cost-aware rewiring, ensuring more optimal paths.
    
- **CMN-RRT* (Central Multi-Node Rapidly-exploring Random Trees)**: Extends RRT* by using multiple root nodes (agents) simultaneously expanding toward the goal. When any tree reaches the goal, intersecting trees connect, significantly reducing redundant exploration and improving path optimality and robustness.
    

Use one of the following algorithm identifiers in your command:

1. `rrt`
    
2. `rrt_star`
    
3. `cmn_rrt_star`
    

### Available Maps

The algorithms can be benchmarked on five predefined map types designed to evaluate performance in diverse environments:

1. `empty` (baseline environment without obstacles)
    
2. `diagonal` (map with diagonal barrier and narrow passages)
    
3. `labyrinth` (maze-like corridors)
    
4. `rectangles` (random rectangular obstacles)
    
5. `random_polygons` (irregular polygon-shaped obstacles)
    

### Example Usage

Run an algorithm using the following command format:

```bash
./main.py --num_agents 1 --map_size 100 100 --step_size 1.0 --map_type labyrinth --algorithm rrt --live_plot
```

Remove the `--live_plot` argument for faster execution without visualisation.

### Benchmarking and Performance

The implemented algorithms were benchmarked on all five maps, evaluating:

- **Success Rate** (finding a valid path)
    
- **Planning Time** (computation speed)
    
- **Path Length** (optimality and efficiency)
    

CMN-RRT* consistently demonstrated the shortest path lengths across all environments, outperforming both RRT and RRT* in robustness and path optimality, especially in complex scenarios.

### Algorithm Examples

#### RRT

![rrt](https://github.com/user-attachments/assets/87ff81af-5170-42ed-9220-4958365646e0)

#### RRT*

![rrt_star](https://github.com/user-attachments/assets/c5d4883a-0271-4c5a-b31a-70620f57a707)

#### CMN-RRT***
