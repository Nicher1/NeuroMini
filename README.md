# NeuroMini



This is how to run the RRT algorithm. You can select between three diferent implementations:
1. rrt
2. rrt_star
3. cmn_rrt_star

And three different maps
1. empty
2. diagonal
3. labyrinth
```bash
./main.py --num_agents 1 --map_size 100 100 --step_size 1.0 --map_type labyrinth --algorithm rrt --live_plot
```
You should remove **_--live_plot_** otherwise the speed will be impacted by the animation function.

