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

For each implmentation, as the number of boids increases, there is a big drop in performance. This is due to the fact that for each boid's velocity computation, we need loop through a list of potential neighbors that can affect the boid. The biggest optimization made in each implementation is a improvement on how efficient we can iterate through these neighbors by partitioning them into a uniform grid and rearrange their data in a coherent memory.

![alt text](https://github.com/trungtle/Project1-CUDA-Flocking/blob/master/images/charts/coherent_performance.png "Coherent uniform grid performance")

Changing the block size increases the number of threads per block. This seems to affects the performance of the coherent uniform grid implementation. It might be due to the fact that more threads can now access data sequentially in a shared warp compared to the other implementations. Increasing the number of block count seems to have the opposite effect as it reduces the number of threads per block instead and data are more scattered through more blocks in memory now.

For the coherent uniform grid, I did see a great performance improvements as the number of boids increases up to 100000 and more. At this point, we immediately see the coherent data pays off. From my performance analysis, I observed that at 150000 boids, coherent uniform grid is ~400ms faster than scattered uniform grid. This is a big improvement!

![alt text](https://github.com/trungtle/Project1-CUDA-Flocking/blob/master/images/charts/coherent_vs_uniform.png "Coherent vs scattered uniform grid performance")


**Note**: I also updated the -arch=sm_52 version to make it compatible for my machine
