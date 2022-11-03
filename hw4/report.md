# HW4

## Problem 1

### Structure 1:

n=5, arch$2\times250$

![](./data/hw4_q1_cheetah_n5_arch2x250_cheetah-cs285-v0_29-10-2022_21-46-39/itr_0_predictions.png)

### Structure 2:

n=500, arch$1\times32$

![](./data/hw4_q1_cheetah_n500_arch1x32_cheetah-cs285-v0_29-10-2022_21-41-42/itr_0_predictions.png)

### Structure 3:

![](./data/hw4_q1_cheetah_n500_arch2x250_cheetah-cs285-v0_29-10-2022_21-41-58/itr_0_predictions.png)

The first structure performs the best. It uses relatively smaller train steps and larger size and number of layers. It makes it less likely to be overfitting. Larger size somehow guarantees the size of features so that it can describe the true features better.



## Problem 2

![](./data/hw4_q2_obstacles_singleiteration_obstacles-cs285-v0_29-10-2022_22-46-00/itr_0_predictions.png)

![](q2.png)

## Problem 3

### Obstacles:

![](./q3_obstacles.png)

### Reacher

![](./q3_reacher.png)

### Cheetah

![](./q3_cheetah.png)



## Problem 4

### Different horizons:

![](./q4_1.png)

Lower horizon better

### Different numseqs:

![](q4_2.png)

Higher numseq better

### Different ensembles:

![](q4_3.png)

ensemble=3 better at last

## Problem 5

![](q5_1.png)

Performance: cem4 > cem2 > random

## Problem 6

![](q6.png)

The one with rollout=10 performs the best.

Hence, larger rollout will lead to better performance.