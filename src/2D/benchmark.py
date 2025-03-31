#!/usr/bin/env python3
import os
import csv
import argparse
import statistics
import numpy as np
import matplotlib.pyplot as plt
from algorithms.rrt_2D import RRT
from algorithms.rrt_star_2D import RRTStar
from algorithms.rrt_star_cmn_agent_class_2D import CMNRRTStarV2
from tqdm import tqdm
from utils.plotter import *
from utils.generator import *
import json

from datetime import datetime

def run_planner(algorithm_cls, args, is_multi_agent=False):
    start = [10, 10]
    goal = [args.map_size[0] - 10, args.map_size[1] - 10]

    planner = algorithm_cls(
        start,
        goal,
        args.num_obstacles,
        args.num_agents if is_multi_agent else 1,
        args.map_size,
        step_size=args.step_size,
        max_iter=10000,
        live_plot=False,
        map_type=args.map_type,
    )
    
    planner.plan()
    return planner

def save_json(results, args):
    results_dir = os.path.join("results", args.map_type)
    os.makedirs(results_dir, exist_ok=True)

    filepath = os.path.join(results_dir, "benchmark_results.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

def save_csv(results, args, filename="benchmark_results.csv"):
    filename = f"benchmark_{args.num_agents}agents_{args.num_obstacles}obs.csv"


    results_dir = os.path.join("results", args.map_type)
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "Run", "Map Type", "Planning Time", "Path Length", "Success"])
        for algo, data in results.items():
            for i in range(len(data["times"])):
                writer.writerow([
                    algo,
                    i + 1,
                    args.map_type,
                    data["times"][i],
                    data["lengths"][i],
                    data["successes"][i]
                ])
    save_json(results, args)

def benchmark(algorithm_name, cls, args, map_obstacles, runs=10, is_multi_agent=False):
    times = []
    lengths = []
    successes = []

    print(f"\nRunning {algorithm_name.upper()} for {runs} runs:")
    

    for i in tqdm(range(runs), desc=f"{algorithm_name.upper()} Benchmark"):
        planner = cls(
            [10, 10],
            [args.map_size[0] - 10, args.map_size[1] - 10],
            args.num_obstacles,
            args.num_agents if is_multi_agent else 1,
            args.map_size,
            step_size=args.step_size,
            max_iter=10000,
            live_plot=False,
            map_type=args.map_type,
            map_obstacles=map_obstacles
        )

        planner.plan()
        save_path_plot(planner, algorithm_name, i, output_dir="results/" + args.map_type + "/" + algorithm_name)

        # Collect results
        times.append(planner.total_planning_time)

        if is_multi_agent:
            agent_paths = [a.results["path_length"] for a in planner.agents if a.goal_reached]
            if agent_paths:
                lengths.append(sum(agent_paths) / len(agent_paths))
                successes.append(1)
            else:
                lengths.append(float("inf"))
                successes.append(0)
        else:
            path_len = planner.compute_path_length(planner.agents[0].path)
            lengths.append(path_len)
            successes.append(1 if planner.agents[0].goal_reached else 0)

    # Close any open figures
    plt.close('all')

    return {
        "times": times,
        "lengths": lengths,
        "successes": successes,
        "time_stats": {
            "mean": statistics.mean(times),
            "std": statistics.stdev(times),
            "min": min(times),
     
            "max": max(times),
        },
        "length_stats": {
            "mean": statistics.mean(lengths),
            "std": statistics.stdev(lengths),
            "min": min(lengths),
            "max": max(lengths),
        },
        "success_rate": 100 * sum(successes) / runs
    }



def plot_benchmark_stats(results, save_path="benchmark_summary.png"):
    algorithms = list(results.keys())

    times_mean = [np.mean(results[a]["times"]) for a in algorithms]
    times_std = [np.std(results[a]["times"]) for a in algorithms]

    lengths_mean = [np.mean(results[a]["lengths"]) for a in algorithms]
    lengths_std = [np.std(results[a]["lengths"]) for a in algorithms]

    x = np.arange(len(algorithms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, times_mean, width, yerr=times_std, label="Planning Time (s)", color='skyblue')
    ax.bar(x + width/2, lengths_mean, width, yerr=lengths_std, label="Path Length", color='lightgreen')

    ax.set_ylabel("Average Value")
    ax.set_title("Benchmark Results (Mean ± StdDev)")
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algorithms])
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)

def write_stats(algorithm_name, result, args, file=None):
    lines = []
    lines.append(f"\n=== {algorithm_name.upper()} Benchmark Results ===")
    lines.append(f"Map Type         : {args.map_type}")
    lines.append(f"Map Size         : {args.map_size[0]} x {args.map_size[1]}")
    lines.append(f"Step Size        : {args.step_size}")
    lines.append(f"Obstacles        : {args.num_obstacles}")
    lines.append(f"Agents           : {args.num_agents if algorithm_name == 'cmn_rrt_star' else 1}")
    lines.append(f"Runs             : {args.num_runs}")
    lines.append(f"Success Rate     : {result['success_rate']:.2f}%")
    lines.append(f"\nPlanning Time (s):")
    lines.append(f"  Mean ± Std     : {result['time_stats']['mean']:.2f} ± {result['time_stats']['std']:.2f}")
    lines.append(f"  Min / Max      : {result['time_stats']['min']:.2f} / {result['time_stats']['max']:.2f}")
    lines.append(f"\nPath Length:")
    lines.append(f"  Mean ± Std     : {result['length_stats']['mean']:.2f} ± {result['length_stats']['std']:.2f}")
    lines.append(f"  Min / Max      : {result['length_stats']['min']:.2f} / {result['length_stats']['max']:.2f}")
    nps = compute_nps(result)
    lines.append(f"NPS Score        : {nps:.4f}")

    text = "\n".join(lines)

    if file:
        print(text, file=file)
    else:
        print(text)

def compute_nps(result, alpha=1.0, beta=0.01):
    time = result["time_stats"]["mean"]
    path = result["length_stats"]["mean"]
    success = result["success_rate"] / 100.0  # normalize

    return success / (1 + alpha * time + beta * path)

def compute_nps_individual(algorithm_name, result, all_results, args, base_alpha=1.0, base_beta=1.0):
    # Get all means for normalization
    all_times = [res["time_stats"]["mean"] for res in all_results.values()]
    all_lengths = [res["length_stats"]["mean"] for res in all_results.values()]

    t_min, t_max = min(all_times), max(all_times)
    l_min, l_max = min(all_lengths), max(all_lengths)

    time = result["time_stats"]["mean"]
    length = result["length_stats"]["mean"]
    success = result["success_rate"] / 100.0  # normalize to [0, 1]

    # Normalize time and length
    norm_time = (time - t_min) / (t_max - t_min + 1e-6)
    norm_length = (length - l_min) / (l_max - l_min + 1e-6)

    # Complexity-scaled α and β
    complexity = 1 + args.num_obstacles / 10
    alpha = base_alpha * complexity
    beta = base_beta * complexity

    # Final NPS
    nps = success / (1 + alpha * norm_time + beta * norm_length)
    return nps



def main():
    parser = argparse.ArgumentParser(description="Benchmark RRT algorithms")
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--map_size", nargs=2, type=int, default=[200, 200])
    parser.add_argument("--step_size", type=float, default=1.0)
    parser.add_argument("--map_type", choices=["empty", "diagonal", "labyrinth", "rectangles", "random_polygons"], default="empty")
    parser.add_argument("--num_obstacles", type=int, default=10)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    results_dir = os.path.join("results", args.map_type)
    os.makedirs(results_dir, exist_ok=True)

    summary_file = os.path.join(results_dir, "summary.txt")
    
    map_obstacles = generate_map(args.map_type, args.map_size, args.num_obstacles)

    results = {}
    results["rrt"] = benchmark("rrt", RRT, args, map_obstacles, args.num_runs)
    with open(summary_file, "w") as f:
        for name in results:
            write_stats(name, results[name], args, file=f)
    write_stats(name, results["rrt"], args)

    # With one agent, CMN RRT * is actually RRT *
    results["rrt_star"] = benchmark("rrt_star", RRTStar, args, map_obstacles, args.num_runs, map_obstacles)
    with open(summary_file, "w") as f:
        for name in results:
            write_stats(name, results[name], args, file=f)
    write_stats(name, results["rrt_star"], args)

    results["cmn_rrt_star"] = benchmark("cmn_rrt_star", CMNRRTStarV2, args, map_obstacles, args.num_runs, is_multi_agent=True)
    with open(summary_file, "w") as f:
        for name in results:
            write_stats(name, results[name], args, file=f)
    write_stats(name, results["cmn_rrt_star"], args)

    save_csv(results, args)
    plot_benchmark_stats(results)

    # Compute individual NPS
    for name, res in results.items():
        nps = compute_nps_individual(name, res, results, args)
        print(f"NPS ({name.upper()}): {nps:.4f}")

if __name__ == "__main__":
    main()
