**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Daniel Krupka
* Tested on: Debian testing (stretch), Intel(R) Core(TM) i7-4710HQ CPU @ 2.50GHz 8GB, GTX 850M

# Project 1 - Boids
This project's goal is to implement [Boids](https://en.wikipedia.org/wiki/Boids) using CUDA,
and to explore a few optimizations that can be made to the naive algorithm. To summarize,
the Boids algorithm implements flocking behavior as seen in birds as an emergent behavior
from a few simple rules. The algorithm is embarassingly parallel, as the behavior of each
Boid depends only on the previous state of the system.

![alt text](images/boid_demo "Boids Demo")
