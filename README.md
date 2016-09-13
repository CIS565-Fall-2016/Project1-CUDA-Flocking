**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

### Performance Analysis
##### 1 - Brute Force
* SM 3.0 Quadro K420
* 5000 - 43-44fps
* 7500 - 20.5 - 21.0fps
* 10000 - 12fps
* 12500 - 6-8fps
* 15000 - 4-5fps
For each implementation, how does changing the number of boids affect performance? Why do you think this is?
For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
---
##### 1 - Uniform Spatial Grid
* 5000 - 81fps (5s), 71fps (10s)
* 7500 - 54fps (5s), 48fps (10s)
* 10000 - 40fps, 36fps
* 15000 - 27, 24
* 20000 - 20, 19
* 25000 - 16
* 30000 - 13
* 40000 - 9.4
* 50000 - 7.1
* 75000 - 3.9
* 100000 - 2.6
* 200000 - 0.9
---
##### 1 - Coherent Data
* 5000 - 84, 80
* 7500 - 55, 54
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