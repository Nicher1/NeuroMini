@echo off
set AGENTS=6
set MAP_SIZE=150 150
set STEP_SIZE=5.0
set OBSTACLES=12
set RUNS=10

setlocal enabledelayedexpansion

:: Define array of map types
set MAP_TYPES=empty diagonal labyrinth rectangles random_polygons

for %%M in (%MAP_TYPES%) do (
    echo ▶ Running benchmark for map: %%M
    python benchmark.py ^
        --num_agents %AGENTS% ^
        --map_size %MAP_SIZE% ^
        --step_size %STEP_SIZE% ^
        --num_obstacles %OBSTACLES% ^
        --map_type %%M ^
        --num_runs %RUNS%
    echo ✅ Done with %%M
    echo -------------------------------
)

echo 🎉 All benchmarks completed!
pause
