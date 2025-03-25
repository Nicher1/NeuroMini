#!/usr/bin/env python3

import argparse
from algorithms.rrt_2D import RRT
from algorithms.rrt_star_2D import RRTStar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Different implementations of RRT Algorithm")

    parser.add_argument("--num_agents", type=int, default=1,
                        help="Number of agents to plan paths for")
    parser.add_argument("--num_obstacles", type=int, default=15,
                        help="Number of wall segments (ignored if using fixed wall layout)")
    parser.add_argument("--map_size", nargs=2, type=int, default=[200, 200],
                        help="Map size as width height")
    parser.add_argument("--step_size", type=float, default=1.0,
                        help="Step size for each tree expansion")
    parser.add_argument("--live_plot", action="store_true",
                    help="Enable live tree drawing during planning")
    parser.add_argument("--map_type", choices=["empty", "diagonal", "labyrinth"], default="empty",
                    help="Choose the type of map layout")

    args = parser.parse_args()

    # Auto-generate start/goal positions (evenly spaced)
    start_positions = [[5 + i * 5, 5] for i in range(args.num_agents)]
    goal_positions = [[args.map_size[0] - 10, args.map_size[1] - 10] for i in range(args.num_agents)]

    cmn_rrt_star = RRT(
        start_positions,
        goal_positions,
        args.num_agents,
        args.num_obstacles,
        args.map_size,
        step_size=args.step_size,
        max_iter=10000,
        live_plot=args.live_plot,
        map_type=args.map_type
    )


    cmn_rrt_star.plan()
    cmn_rrt_star.animate()