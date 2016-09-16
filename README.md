**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Trung Le
* Windows 10 Home, i7-4790 CPU @ 3.60GHz 12GB, GTX 980 Ti (Personal desktop)

### Flocking

#### Description

This project implemented three different algorithms for flocking behavior in CUDA: brute-force, uniform grid, coherent uniform grid.

![alt text](https://github.com/trungtle/Project1-CUDA-Flocking/blob/master/images/screenshots/Uniform_5000_boids.gif "Flocking simulation")

**---- General information for CUDA device ----**
- Device name: GeForce GTX 980 Ti
- Compute capability: 5.2
- Compute mode: Default
- Clock rate: 1076000
- Integrated: 0
- Device copy overlap: Enabled
- Kernel execution timeout: Enabled
 
**---- Memory information for CUDA device ----**

- Total global memory: 6442450944
- Total constant memory: 65536
- Multiprocessor count: 22
- Shared memory per multiprocessor: 98304
- Registers per multiprocessor: 65536
- Max threads per multiprocessor: 2048
- Max grid dimensions: [2147483647, 65535, 65535]
- Max threads per block: 1024
- Max registers per block: 65536
- Max thread dimensions: [1024, 1024, 64]
- Threads per block: 512


#### Performance analysis

To run performance analysis, I used CUDA to wrap around the simulation step and turned off visualization.

The boid count tested are: 1000, 5000, 50000, 100000, 150000
The block size tested are: 1, 128, 512 (maxed at 512 threads per block on my machine)

**Table showing simulation time per frame (in ms) vs. number of boids). Block size is 128.**

| Boid count |	Coherent	| Uniform |	Naive|
| ---------- | -------- | ------- | ---- |
|5000|	0.2|	0.2|	4.3|
|50000|	1.2|	1.5|	166|
|100000|	1.8|	2.6|	654|
|500000|	12.5|	20.6|	(crash)|
|1000000|	52|	74|	(crash)|

![alt text](https://github.com/trungtle/Project1-CUDA-Flocking/blob/master/images/charts/performace.png "Naive vs. Coherent vs scattered uniform grid performance")

For each implmentation, as the number of boids increases, there is a big drop in performance. This is due to the fact that for each boid's velocity computation, we need loop through a list of potential neighbors that can affect the boid. The biggest optimization made in each implementation is a improvement on how efficient we can iterate through these neighbors by partitioning them into a uniform grid and rearrange their data in a coherent memory.

Changing the block size doesn't seem to increases performace. However, I did find an interesting pattern. At every block size that is a power of 2, the performance is optimal. I tested this with 1,000,000 in coherent grid (~46ms max) and scattered uniform grid (~74ms max).

**Table showing block size vs. simulation time per frame (in ms)**

| Block size |	Coherent |	Uniform |
| ---------- | -------- | ------- |
|32|	45|	74|
|35|	74|	113|
|64|	44|	73|
|100|	54|	87|
|128|	46|	74|
|256|	45|	73|
|400|	54|	85|
|512|	46|	73|
|600|	68|	105|
|1024|	46|	72|

When comparing between coherent and scattered uniform grids, I did see a great performance improvement with the coherent data implementation as the number of boids increases. From my performance analysis, I observed that at 1000000 boids and kernel block size 128, coherent uniform grid is ~20ms faster than scattered uniform grid. This is a big improvement! This is due to the fact that we take advantage of spatial coherence can access each neighboring boid's data sequentially in memory.

**Note**: I also updated the -arch=sm_52 version to make it compatible for my machine
