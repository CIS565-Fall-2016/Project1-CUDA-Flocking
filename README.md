CIS 565 Project 1 - Flocking
============================
* Richard Lee
* Tested on: Windows 7, i7-3720QM @ 2.60GHz 8GB, GT 650M 4GB (Personal Computer)

![](images/flocking.gif)

## Performance analysis

Performance testing was done by measuring the time taken to run 1000 frames of the simulation on each implementation with the different variables.

**Number of boids**
Testing the number of boids was run with a block size of 128.
![](images/chart1.png)
The time elapsed took longer as the number of boids increased, which was to be expected due to the increased number of comparisons.

**Block size**
Testing the block size was run with 5000 boids.
![](images/chart2.png)
Changing the block size caused fluctuations in the performance of all three methods, but did not lead to any significant increases or decreases in their performance. This could be because the increased number of available threads may not have been efficiently utilized to improve the performance. 

**Coherent uniform grid analysis**
For the coherent uniform grid, the performance was about equal at 5000 agents, with the scattered uniform grid being slightly faster than the coherent grid. However, as the number of agents increased, the time taken for the coherent grid increased at a slower rate and outperformed the scattered grid. 

I expected the coherent grid to outperform the scattered grid at all numbers of agents, and I think this result was because less agents meant that there wasn't as much chance to take advantage of the contiguous boid data, and led to a small difference in performance due to the extra pre-processing step to shuffle the boid data. 

However, the coherent grid allowed for faster direct access of all the boids after the pre-processing shuffle as the number of boids increased, while the scattered grid still had to perform lookups for each boid in the position and velocity arrays, which led to performance advantages as the number of boids scaled up.