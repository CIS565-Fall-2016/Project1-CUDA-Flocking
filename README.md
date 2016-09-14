**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Ottavio Hartman
* Tested on: Windows 7, i7-4790 @ 3.60GHz 16GB, Quadro K420 (Moore 102 Lab)

![alt text]("https://github.com/omh1280/Project1-CUDA-Flocking/raw/master/images/Capture.PNG")

### Performance Analysis

Data is in the form: __# of points - frames per second__
##### 1 - Brute Force
These tests were performed with a grid size of 5.0. 
* 5000 - 43 (5s in) - 44 (10s in)
* 7500 - 20.5 (5s in) - 21.0 (10s in)
* 10000 - 12
* 12500 - 6 (5s in) - 8fps (10s in)
* 15000 - 4 (5s in) - 5fps (10s in)

These tests were performed with a grid size of 10.0
* 5000 - 43.1
* 15000 - 4

###### Results
The brute force method seems to decay in speed very quickly with the number of boids added. There is not much 
decay over time, however, with most of the 5-10 second intervals staying around the same fps. I think this is 
because each thread is doing pretty much the same number of calculations regardless of the positions of 
the neighboring boids. That is, each thread is looping through the entire array of boids. Changing the neighbor
distance from 5.0 to 10.0 had no noticeable effect on this algorithm. That is because it will loop over all of the
boids regardless of the neighbor distance.
---
##### 2 - Uniform Spatial Grid
These tests were performed with a grid size of 5.0.
* 5000 - 81 (5s in), 71 (10s in)
* 7500 - 54 (5s), 48 (10s in)
* 10000 - 40 (5s in), 36 (10s in)
* 15000 - 27 (5s in), 24 (10s in)
* 20000 - 20 (5s in), 19 (10s in)
* 25000 - 16
* 30000 - 13
* 40000 - 9.4
* 50000 - 7.1
* 75000 - 3.9
* 100000 - 2.6
* 200000 - 0.9

These tests were performed with a grid size of 10.0.
* 5000 - 50 (5s in), 41 (10s in)
* 25000 - 7.4 (5s in), 6.8 (10s in)
* 75000 - 1.4

###### Results
The uniform spatial grid _greatly_ improves the performance of this program. As more boids are added, the frames
per second decrease at a much slower rate than the brute force method. This allows this algorithm to reach much
higher boid numbers (like 200,000) at the speed of 0.9 frames per second. However, there is a greater decay in
the speed of the program with regard to time. I think this is because as the boids "group up" there tends to be 
denser cells, which means that each of the boids in a dense cell needs to loop over the other boids in that cell.
This means that as the boids group together the running time of the algorithm would tend towards the brute-force
method.
---
##### 3 - Coherent Data
These tests were performed with a grid size of 5.0.
* 5000 - 84 (5s in), 80 (10s in)
* 7500 - 55 (5s in), 54 (10s in)
* 10000 - 42
* 15000 - 28
* 20000 - 21
* 25000 - 16.8
* 30000 - 13.9
* 40000 - 10.4
* 50000 - 8.3
* 75000 - 5.5
* 100000 - 4.0
* 200000 - 1.9

These tests were performed with a grid size of 10.0.
* 5000 - 75 (5s in), 68 (10s in)
* 25000 - 13.9 (5s in), 12.8 (10s in)
* 75000 - 3.2

###### Results
This algorithm clearer ran the fastest. For low boid numbers (5,000 - 50,000), it performed <10% better than
the uniform spatial grid algorithm. However, it really began to shine as the boid count increased above 50,000.
For 200,000 boids, for example, it is more than twice as fast as the other algorithm. This did surprise me, but it
makes sense: removing one level of indirection in each kernel and implementing contiguous memory accesses are small
efficiencies which add up over time. That is, it only provides major efficiency improvement as the number of 
memory accesses increases a lot.