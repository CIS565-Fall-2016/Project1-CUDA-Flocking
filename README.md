**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

![alt text] (https://github.com/lobachevzky/Project1-CUDA-Flocking/blob/working/project1.gif "Running with all possible optimizations")

* Ethan Brooks
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

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
Since all three flocking rules only apply to boids within a certain proximity, it is possible to cut down the number of comparisons by discretizing the 3d space into cells and only comparing boids within the same cells or adjacent cells.


