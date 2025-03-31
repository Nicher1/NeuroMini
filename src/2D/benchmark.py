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

def benchmark(algorithm_name, cls, args, map_obstacles, is_multi_agent, runs=10):
    times = []
    lengths = []
    successes = []
    iterations = []

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
            map_obstacles=map_obstacles,
            debug=args.debug
        )

        planner.plan()
      
        save_path_plot(planner, algorithm_name, i, output_dir="results/" + args.map_type + "/" + algorithm_name, multi_agent=is_multi_agent)

        # Collect results
      

        # Main Agent - the one that starts at the Starting Node.
        agent = planner.agents[0]

    
    
        if agent.goal_reached:
            successes.append(1)
            lengths.append(agent.results["path_length"])
            iterations.append(agent.results["iterations"])
            times.append(agent.results["time"])
        else:
            successes.append(0)
            lengths.append(float("inf"))
            iterations.append(float("inf"))
            times.append(float("inf"))
                
        # Close any open figures
        plt.close('all')

    return {
        "times": times,
        "lengths": lengths,
        "iterations" : iterations,
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
        "iteration_stats": {
            "mean": statistics.mean(iterations),
            "std": statistics.stdev(iterations),
            "min": min(iterations),
            "max": max(iterations),
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

def save_summary_csv(results, args, filename="benchmark_summary_stats.csv"):
    results_dir = os.path.join("results", args.map_type)
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Algorithm", "Map Type",
            "Planning Time Mean", "Planning Time Std", "Planning Time Min", "Planning Time Max",
            "Path Length Mean", "Path Length Std", "Path Length Min", "Path Length Max",
            "Iterations Mean", "Iterations Std", "Iterations Min", "Iterations Max",
            "Success Rate", "NPS Score"
        ])

        for algo, res in results.items():
            writer.writerow([
                algo,
                args.map_type,
                f"{res['time_stats']['mean']:.2f}", f"{res['time_stats']['std']:.2f}", f"{res['time_stats']['min']:.2f}", f"{res['time_stats']['max']:.2f}",
                f"{res['length_stats']['mean']:.2f}", f"{res['length_stats']['std']:.2f}", f"{res['length_stats']['min']:.2f}", f"{res['length_stats']['max']:.2f}",
                f"{res['iteration_stats']['mean']:.2f}", f"{res['iteration_stats']['std']:.2f}", f"{res['iteration_stats']['min']:.2f}", f"{res['iteration_stats']['max']:.2f}",
                f"{res['success_rate']:.2f}",
                f"{compute_nps(res):.4f}"
            ])

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
    lines.append(f"\nIterations:")
    lines.append(f"  Mean ± Std     : {result['iteration_stats']['mean']:.2f} ± {result['iteration_stats']['std']:.2f}")
    lines.append(f"  Min / Max      : {result['iteration_stats']['min']:.2f} / {result['iteration_stats']['max']:.2f}")
    nps = compute_nps(result)
    lines.append(f"NPS Score        : {nps:.4f}")

    text = "\n".join(lines)

    if file:
        print(text, file=file)
    else:
        print(text)

def compute_nps(result, alpha=1.0, beta=0.01, gamma=0.2):
    time = result["time_stats"]["mean"]
    path = result["length_stats"]["mean"]
    success = result["success_rate"] / 100.0  # normalize
    iterations = result["iteration_stats"]["mean"]

    return success / (1 + alpha * time + beta * path + gamma * iterations)

def compute_nps_individual(algorithm_name, result, all_results, args, base_alpha=1.0, base_beta=1.0, base_gamma=1.0):
    # Get all means for normalization
    all_times = [res["time_stats"]["mean"] for res in all_results.values()]
    all_lengths = [res["length_stats"]["mean"] for res in all_results.values()]
    all_iterations = [res["iteration_stats"]["mean"] for res in all_results.values()]

    t_min, t_max = min(all_times), max(all_times)
    l_min, l_max = min(all_lengths), max(all_lengths)
    i_min, i_max = min(all_iterations), max(all_iterations)

    time = result["time_stats"]["mean"]
    length = result["length_stats"]["mean"]
    iterations = result["iteration_stats"]["mean"]
    success = result["success_rate"] / 100.0  # normalize to [0, 1]

    # Normalize time and length
    norm_time = (time - t_min) / (t_max - t_min + 1e-6)
    norm_length = (length - l_min) / (l_max - l_min + 1e-6)
    norm_iterations = (iterations - i_min) / (i_max - i_min + 1e-6)

    # Complexity-scaled α and β and γ
    complexity = 1 + args.num_obstacles / 10
    alpha = base_alpha * complexity
    beta = base_beta * complexity
    gamma = base_gamma * complexity

    # Final NPS
    nps = success / (1 + alpha * norm_time + beta * norm_length + gamma * norm_iterations)
    return nps



def main():
    print("Benchmark running")
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

    summary_file = os.path.join(results_dir, f"{args.map_type}_summary.txt")
    
    map_obstacles = generate_map(args.map_type, args.map_size, args.num_obstacles)



    results = {}
    results["rrt"] = benchmark("rrt", RRT, args, map_obstacles, False, args.num_runs)
    with open(summary_file, "w") as f:
        for name in results:
            write_stats(name, results[name], args, file=f)
    write_stats(name, results["rrt"], args)
    

    # With one agent, CMN RRT * is actually RRT *
    results["rrt_star"] = benchmark("rrt_star", RRTStar, args, map_obstacles, False, args.num_runs)
    with open(summary_file, "w") as f:
        for name in results:
            write_stats(name, results[name], args, file=f)
    write_stats(name, results["rrt_star"], args)

    results["cmn_rrt_star"] = benchmark("cmn_rrt_star", CMNRRTStarV2, args, map_obstacles, True, args.num_runs)
    with open(summary_file, "w") as f:
        for name in results:
            write_stats(name, results[name], args, file=f)
    write_stats(name, results["cmn_rrt_star"], args)

    save_csv(results, args)
    save_summary_csv(results, args)
    plot_benchmark_stats(results)

    # Compute individual NPS
    for name, res in results.items():
        nps = compute_nps_individual(name, res, results, args)
        print(f"NPS ({name.upper()}): {nps:.4f}")

if __name__ == "__main__":
    main()
