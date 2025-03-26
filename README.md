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

### RRT
![rrt](https://github.com/user-attachments/assets/87ff81af-5170-42ed-9220-4958365646e0)

### RRT*
![rrt_star](https://github.com/user-attachments/assets/c5d4883a-0271-4c5a-b31a-70620f57a707)



