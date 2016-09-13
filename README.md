**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Bowen Bao
* Tested on: Windows 10, i7-6700K @ 4.00GHz 32GB, GTX 1080 8192MB (Personal Computer)

## Overview

Here's the simulation of both 5000 and 50000 boids:

![Simulation of 5000 boids](/images/boid_5000.gif)

![Simulation of 50000 boids](/images/boid_50000.gif)

## Instructions to run
Most of my code structures stay the same as the original skeleton code. One slight change is that one has to remember to also change COHERENT_GRID in kernel.cu depending on if they are simulating in uniform grid coherent or not, instead of only changing the COHERENT_GRID in main.cpp.  


## Performance Analysis
### Different number of boids

These tests are run with the block size of 128. Each simulation is roughly 15 seconds long. Here in the following graphs, we measure the performance of the device function that updates the velocity of every boid.

![](/images/boid_plot_1.png) ![](/images/boid_plot_2.png)

Observe that uniform grid methods greatly outperforms the naive solution, as the number of neighbor boids each boid needs to check is greatly reduced. We could also observe that uniform grid with sorted boid position and velocity has a better performance. This is a trade-off between the additional overhead of sorting the boid data, and the performance gain of being able to sequentially access the memory while calculating boid velocity. In this case, the benefit outweighs the cost. In fact, we could observe that the average cost of sorting data is very low (<0.1ms) compared to the cost of calculating velocity (~2ms) for 200,000 boids.

### Different block size

The following tests each run with 50000 boids. Each simulation is roughly 15 seconds long.

![](/images/blocksize_plot_1.png) ![](/images/blocksize_plot_2.png)

We could observe that the performance varies slightly with larger block size. This probably is due to that in this problem, each thread is completely independent of other threads.

## Questions

### For each implementation, how does changing the number of boids affect performance? Why do you think this is?
More boids leads to worse performance. This is expected as the naive solution has a complexity of O(n^2) of updating the velocity, where n is the number of boids. And the other two methods has informally a complexity of O(nm), where m is a number that in average greatly smaller than n, but in worst case could be n. 

### For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
As mentioned in Section Different block size. Changing block count and block size didn't affect performance very much. As in this problem no thread is waiting on any other threads.

### For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
Yes. In the naive uniform grid, while updating each boid velocity, it needs to jump around in memory to fetch the data of the boid's neighbors. In coherent uniform grid however, the performance benefits from sequentially accessing the memory for the data of neighbor boids. Since the complexity of calculating the velocity is still polynomial, it would seem that a O(nlogn) sorting complexity overhead wouldn't become a very large problem. And my performance analysis supports this expectation.