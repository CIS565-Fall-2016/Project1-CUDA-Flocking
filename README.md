**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

![alt text] (https://github.com/lobachevzky/Project1-CUDA-Flocking/blob/working/project1.gif "Running with all possible optimizations")

* Ethan Brooks
* Tested on: Windows 7, Intel(R) Xeon(R), GeForce GTX 1070 8GB (SIG Lab)

#Flocking Simulation

## Summary
Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

This project simulates flocking behavior by graphically depicting "boids", or colored points,  in a 3-d environment which obey three rules:

1. Boids try to fly towards the centre of mass of neighbouring boids.
2. Boids try to keep a small distance away from other objects (including other boids).
3. Boids try to match velocity with near boids.

These rules cause groups of boids to coalesce into flocks, with all the boids in a flock flying in parallel, close to one another, at similar velocities.

## Optimizations.
This project includes three implementations: 
1. A naive one that compares each boid to every other boid
2. One that only compares boids within neighborhoods.
3. A further optimization of 2 that minimizes memory access.

### Implementation 1
Though this implementation does utilize the GPU, calling kernel functions on each boid which are executed in parallel, each of these kernel functions compares its assigned boid with every other boid in the simulation. Thus every kernel function is O(n), where n is the number of boids.

### Implementation 2
Since all three flocking rules only apply to boids within a certain proximity, it is possible to cut down the number of comparisons by discretizing the 3d space into cells and only comparing boids within the same cells or adjacent cells. Our implementation maps boids to cells based on their (x, y, z) positions. Therefore, given a boid and its position, we can easily identify the id of the cell in which it resides. We subsequently identify cells adjacent to this one in all three directions (27 total). 

The next task is to identify the boids within each of these 27 cells. A naive approach would search all boids in the simulation and identify those within the 27 cells. However, this would defeat the purpose since we would be back to linear time complexity as in Implementation 1. In order to avoid this, we develop a second buffer of pointers to boids, sorted by cell location and then map each cell to a range within this buffer. This way, given a cell, we use this map to identify the range within the second buffer to search. By following the pointers within this second buffer, we can access the boids that are within the cell.

Our time complexity for each kernel invocation is still O(n), but now n is the number of boids in the neighboring cells, not in the entire simulation -- a small fraction of the total number. In practice, as the chart below depicts, the number of boids in neighboring cells remains relatively constant and the execution only increases a little as the number of boids increases.

### Implementation 3
The previous approach has one major weakness: each cell is mapped to an array of _pointers_ to boids. Therefore when searching within each cell, we actually have to follow a pointer for each boid, and since a given boid is likely in the vicinity of multiple boids, we actually have to follow the pointer for the same boid repeatedly. On the GPU, memory access is slow. In order to minimize memory access, we instead _sort the boids themselves_--that is, we sort the positions and velocities by cell. To do so, we still develop a second buffer that we sort by cell index, but this time we use that buffer to rearrange the actual positions and velocity buffers themselves.

This implementation actually requires us to resort the boids every time step, since boids change position and do not stay within the same cell necessarily. However sorting is cheap on the GPU--O(log n)--and we only perform this step once per timestep, whereas the memory accesses in Implementation 2 happened repeatedly. 

Another advantage of this approach is that it puts adjacent boids close to each other in memory. In Implementation 2, the boid pointers that we were following might point to any location in the position and velocity buffers and therefore to disparate locations in memory. In contrast, this implementation places the positions and velocities for a given cell be next to each other. This helps make memory access quicker by obeying Locality of Reference, which ensures that memory accesses nearby to previous accesses are quicker.

The memory improvements are evident in the following graph:
![alt text] (https://github.com/lobachevzky/Project1-CUDA-Flocking/blob/master/Performance_Page_1.png "A comparison of performance across implementations.")
In this graph, time per frame is averaged across 1000 frames.

### Optimization across block sizes
Finally, we experimented with different block counts. Each experiment was run with 2^15 boids. The results are shown below:
![alt text] (https://github.com/lobachevzky/Project1-CUDA-Flocking/blob/master/Performance_Page_2.png "A comparison of performance across implementations.")
