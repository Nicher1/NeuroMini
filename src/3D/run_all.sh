#!/bin/bash

AGENTS=6
MAP_SIZE="100 100"
STEP_SIZE=5.0
OBSTACLES=12
RUNS=10  # You can change this if needed

MAP_TYPES=("empty" "diagonal" "labyrinth" "rectangles" "random_polygons")

for MAP in "${MAP_TYPES[@]}"; do
    echo "▶ Running benchmark for map: $MAP"
    ./benchmark.py \
        --num_agents $AGENTS \
        --map_size $MAP_SIZE \
        --step_size $STEP_SIZE \
        --num_obstacles $OBSTACLES \
        --map_type $MAP \
        --num_runs $RUNS
    echo "✅ Done with $MAP"
    echo "-------------------------------"
done

echo "🎉 All benchmarks completed!"