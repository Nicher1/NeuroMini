#!/usr/bin/env python3

import argparse
from algorithms.rrt_2D import RRT
from algorithms.rrt_star_2D import RRTStar
from algorithms.rrt_star_cmn_2D import CMNRRTStar

def main():
    parser = argparse.ArgumentParser(description="Different implementations of RRT Algorithms")

    parser.add_argument("--algorithm", choices=["rrt", "rrt_star", "cmn_rrt_star"], default="rrt",
                        help="Choose which RRT-based algorithm to run")

    parser.add_argument("--num_agents", type=int, default=1,
                        help="Number of agents (only supported in cmn_rrt_star)")
    parser.add_argument("--map_size", nargs=2, type=int, default=[200, 200],
                        help="Map size as width height")
    parser.add_argument("--step_size", type=float, default=1.0,
                        help="Step size for each tree expansion")
    parser.add_argument("--live_plot", action="store_true",
                        help="Enable live tree drawing during planning")
    parser.add_argument("--map_type", choices=["empty", "diagonal", "labyrinth"], default="empty",
                        help="Choose the type of map layout")

    args = parser.parse_args()

    start_positions = [10, 10]
    goal_positions = [args.map_size[0] - 10, args.map_size[1] - 10]

    # Algorithm selection
    if args.algorithm == "rrt":
        planner = RRT(
        start_positions,
        goal_positions,
        args.num_agents,
        args.map_size,
        step_size=args.step_size,
        max_iter=10000,
        live_plot=args.live_plot,
        map_type=args.map_type
        )

    elif args.algorithm == "rrt_star":
        planner = RRTStar(
        start_positions,
        goal_positions,
        args.num_agents,
        args.map_size,
        step_size=args.step_size,
        max_iter=10000,
        live_plot=args.live_plot,
        map_type=args.map_type
        )

    elif args.algorithm == "cmn_rrt_star":
        planner = CMNRRTStar(
        start_positions,
        goal_positions,
        args.num_agents,
        args.map_size,
        step_size=args.step_size,
        max_iter=10000,
        live_plot=args.live_plot,
        map_type=args.map_type
        )

    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    # Run planning
    planner.plan()
    planner.animate()


if __name__ == "__main__":
    main()