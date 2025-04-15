#!/usr/bin/env python3

import argparse
from algorithms.rrt_3D import RRT
from algorithms.rrt_star_3D import RRTStar
from algorithms.rrt_star_cmn_agent_class_3D import CMNRRTStar
from algorithms.rrt_star_cmn_parallel_3D import CMNRRTStarParallel
from utils.generator import generate_map

def main():
    parser = argparse.ArgumentParser(description="Different implementations of RRT Algorithms")

    parser.add_argument("--algorithm", choices=["rrt", "rrt_star", "cmn_rrt_star", "cmn_rrt_star_parallel"], default="rrt",
                        help="Choose which RRT-based algorithm to run")

    parser.add_argument("--num_agents", type=int, default=1,
                        help="Number of agents (only supported in cmn_rrt_star)")
    parser.add_argument("--map_size", nargs=3, type=int, default=[200, 200, 50],
                        help="Map size as width height")
    parser.add_argument("--step_size", type=float, default=1.0,
                        help="Step size for each tree expansion")
    parser.add_argument("--live_plot", action="store_true",
                        help="Enable live tree drawing during planning")
    parser.add_argument("--map_type", choices=["empty", "diagonal", "labyrinth", "rectangles"], default="empty", #, "random_polygons"], default="empty",
                        help="Choose the type of map layout")
    parser.add_argument("--num_obstacles", type=int, default=10,
                    help="Number of obstacles (used for random maps)")
    parser.add_argument("--z_height", type=int, default=10, help="Z height for 3D map obstacles")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug")

    args = parser.parse_args()

    start_positions = [10, 10, 10]
    goal_positions = [args.map_size[0] - 10, args.map_size[1] - 10, args.z_height // 2]
    print(f"# ------- Running {args.algorithm.upper()} ------- #")
    common_args = {
        "start_position": start_positions,
		"goal_position": goal_positions,
        "num_obstacles": args.num_obstacles,
        "num_agents": args.num_agents,
        "map_size": args.map_size,
        "step_size": args.step_size,
        "max_iter": 10000,
        "live_plot": args.live_plot,
        "map_type": args.map_type,
        "z_height": args.z_height
    }
    # Algorithm selection
    if args.algorithm == "rrt":
        planner = RRT(**common_args)
    elif args.algorithm == "rrt_star":
        planner = RRTStar(**common_args)
    elif args.algorithm == "cmn_rrt_star":
        planner = CMNRRTStar(**common_args, debug=args.debug)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    # Run planning
    planner.plan()
    # final_path = planner.agents[0].path  # this is the full list of (x, y, z)
    planner.animate()


if __name__ == "__main__":
    main()
