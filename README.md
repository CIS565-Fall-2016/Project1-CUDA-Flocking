#University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1 - Flocking#

## Xiaomao Ding ##
* Tested on: Windows 8.1, i7-4700MQ @ 2.40GHz 8.00GB, GT 750M 2047MB (Personal Computer)

## Intro ##
The code in this repo is part of Project 1 for CIS565 Fall 2016 at UPenn. For this project, I accelerated the [Reynolds Boid algorithm](http://www.red3d.com/cwr/boids/) using a NVIDIA Cuda kernel. There are three different implementations: a brute force method that compares each boid to every other boid, a uniform grid method that divides the space into a grid for neighbor search, and a coherent uniform grid method that reorganizes the position and velocity vectors to represent the uniform grid. See the gif below for an example of the algorithm in action! Each color represents a different flock of boids. 

<div style="text-align:center"><img src ="https://github.com/xnieamo/Project1-CUDA-Flocking/blob/master/images/dt0.2_particles16000.gif" /></div>

Above is a gif generated using the code in this repo with 16000 boids using the coherent grid implementation.

### Quick Note ###
Before running any of the code in this repo, it is possible that you may have to adjust the compute capability flag in `scr/CMakeLists.txt`. To do so, change the '-arch=sm_30' to match your compute capability. 20 matches to 2.0, 30 to 3.0, etc.

## Performance Analysis ##

### Number of Boids ###
In order to analyse the performance of our implementation, we will be using the FPS without visualization as a metric. The first thing we would like to know is how well each algorithm performs with the number of boids.

![FPSvNumBoidPlot](https://github.com/xnieamo/Project1-CUDA-Flocking/blob/master/images/PerformanceVBoidNum.png)

In the graph above, it is clear that the brute force method performs the worst. This is because the number of comparisons for each boid increases linearly with the number of boids. The scattered uniform grid performs much better as it drastically reduces the number of comparisons needed for each boid. What's surprising is the dramatic increase in performance for the uniform grid! Even though both the coherent and scattered grid make the same number of comparisons, the difference in performance is similar to that between the scattered grid and the brute force method.  The only change is that we remove the use of an intermediate array for grid indexing. This suggests that reading from memory is a significant bottleneck in our GPU implementations.

### Block Size and Count###
We might also be interested in the performance of our implementations for varying block sizes on the GPU. Below we see that performance is roughly equivalent for each implementation. This makes sense as increasing the block sizes (and thereby decreasing block count) doesn't necessarily change the number of threads allocated for the whole calculation and each boid has an independent calculation. These graphs were generated using a coherent grid with 16000 boids.

![FPSvBlockSize](https://github.com/xnieamo/Project1-CUDA-Flocking/blob/master/images/PerformanceVBlockSize.png)

### dT ###
What's somewhat surprising is that changing the time step parameter, dT, also affects performance. As you increase dT, performance increases drastically as shown in the graph below. This is possibly related to the fact that at high dT, all the boids are automatically placed into 1 giant flock. This graph is generated using a coherent grid with 16000 boids.

![FPSvdT](https://github.com/xnieamo/Project1-CUDA-Flocking/blob/master/images/PerformanceVdt.png)





