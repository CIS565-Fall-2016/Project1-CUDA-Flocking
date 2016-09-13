

![](images/boids_meme.jpg)

Implemented by: [Gabriel Naghi](https://www.linkedin.com/in/gabriel-naghi-78ab4738) on Windows 7, i7-xxx @ xGHz xxGB, Quadro K620 2048MB (Moore 100C Lab)

University of Pennsylvania, [CIS 565: GPU Programming and Architecture](http://www.seas.upenn.edu/~cis565/)


Project 1 - Flocking
=====================

![](images/simulation.png)

Over the course of this project, we implemented a flocking simulation. It is inteded to mimic roughly the behavior of groups of fish or birds- known throughout the code base as Boids.

There are 3 components to the flocking algorithm:
1. Boids gravitate toward the local center of gravity within a radius r1. 
2. Boids maintain a minimum disance r2 from their neighbors.
3. Boids attmpt to match the velocity of their neighbors within a radius r3.

We implemented three different methods of calculating the effects of these rules. The first, the naive implementation, checks, for each boid, every other boid and applies each of the rules if they are within the area of effect. The second implementation utilized a uniform grid which sorted the boid indices by the "sector" of the scene they occupied, and only checked the relevant adjacent cells for boids. The final implementation also udes a uniform grid, but removed one layer of indirection by resorting the data itself rather than saving a pointer to its original location, maximizing data coherence.

Perfomance Analysis
----------------------

Naive Implementation



Uniform Grid Implementation



Coherent Grid Implementation

