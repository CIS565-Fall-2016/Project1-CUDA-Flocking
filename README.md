**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Name: Zhan Xiong Chin
* Tested on: Windows 7 Professional, Intel(R) Xeon(R) CPU E5-1630 v4 @ 3.70 GHz 3.70 GHz, GTX 1070 8192MB (SIG Lab)

![](images/boids2.gif)

10000 particle simulation

![](images/boids.gif)

500000 particle simulation


Build Instructions
==================
[See here](https://github.com/CIS565-Fall-2016/Project0-CUDA-Getting-Started/blob/master/INSTRUCTION.md)

Performance analysis
====================

|                        | 5000 particles | 10000 particles | 50000 particles | 500000 particles |
|------------------------|----------------|-----------------|-----------------|------------------|
| Naive search           | 700 fps        | 350 fps         | 20 fps          | (crashes)        |
| Scattered uniform grid | 770 fps        | 1100 fps        | 490 fps         | 6 fps            |
| Coherent uniform grid  | 770 fps        | 1100 fps        | 1100 fps        | 72 fps           |

* For the naive search, increasing the number of boids decreases performance, which is expected, as more computations
 need to be done to figure out velocity changes when the number of boids increases.
* However, for the uniform grids, the performance increases when going from 5000 to 10000 particles, but decreases
after that. This may be related to branching effects: there may not be empty grid cells with a sufficient number of particles.

* Increasing the block size does not change the performance of the uniform grid implementation significantly. This is
because the overall utilization of the device remains similar.

* The coherent uniform grid had significant performance improvement over the scattered uniform grid. This was expected,
since this takes better advantage of caching and avoids expensive memory accesses.