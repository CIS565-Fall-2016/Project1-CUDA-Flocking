
####University of Pennsylvania
####CIS 565: GPU Programming and Architecture

##Project 1 - Flocking

* Xueyin Wan
* Tested on: Windows 10, i7-4870 @ 2.50GHz 16GB, NVIDIA GeForce GT 750M 2GB (Personal Laptop)

==================================================================
###Final Result Screenshot
![alt text](https://github.com/xueyinw/Project1-CUDA-Flocking/blob/master/images/Xueyin_Performance.gif "Xueyin's Performance Analysis")

####Parameters:
* Number of boids = 15000
* dT = 0.2
* Algorithm used in the screenshot : Coherent Uniform Grid
* BlockSize = 128
* rule1Distance  = 5.0f,  rule1Scale = 0.01f
* rule2Distance = 3.0f, rule2Scale = 0.1f
* rule3Distance = 5.0f, rule3Scale = 0.1f
* maxSpeed = 1.0f
* scene_scale = 100.0f

==================================================================
###Performance Analysis


I choose to use 1st method : Disable visualization (#define VISUALIZE to 0 ) to  measure performance.
###Without Visualization
####(#define VISUALIZE 0)
|    Number of boids | 5000 | 15000 | 25000 | 35000 | 45000 | 55000 | 65000 | 75000 | 85000 | 95000 |
| ------------- |:-------------:| -----:|
| Brute Force neighbor search FPS | 57.7 | 6.6 | 2.2 | | | | | | | |
| Uniform Grid neighbor search  FPS  | 580 | 250 | 160 | 108.4 | 80.4 | 63.6 | 53.2 | 42.7 | 30.5 | 25.7 |  
| Coherent Uniform Grid neighbor search FPS | 680 | 300 | 180 | 130 | 100.7 | 78.3 | 67.4 | 57.4 | 49.5 | 39.7 |

We could see the result from this visualized chart I made.
![alt text](https://github.com/xueyinw/Project1-CUDA-Flocking/blob/master/images/AlgorithmComparision.png"Xueyin's Updated Chart")

We could see the comparison of the FPS situation between Brute Search, Uniform Grid and Coherent Uniform Grid when boids' number increases.

###Questions & Answer
####1. For each implementation, how does changing the number of boids affect performance? Why do you think this is?
Answer:

* Brute Force neighbor search algorithm: as the number of boids increases, frame-rate decreases very fast
* Uniform Grid neighbor search: the number of boids could as many as almost  80000 as the fps keeps at 60, performance is much better than  Brute Force neighbor search algorithm.
* Coherent Uniform Grid neighbor search: the number of boids could as many as almost 100000 as the fps keeps at 60, performance is much better than Brute Force neighbor search algorithm and a little better than Uniform Grid neighbor search.



####2.For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

Answer:

* Generally speaking, when block count decreases and block size increases , the performance will be better.
* But in order to get a great performance, we should make a balance between block count and block size, and set their value wisely in order to improve memory performance.

####3. For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
Answer:

* My answer is yes. As my first two tables at Performance Analysis part, we can see that Coherent Uniform Grid neighbor search is better than Uniform Grid neighbor search. When writing codes to implement Coherent Uniform Grid neighbor search in part 2.3 , I rearranged the boid data itself so that all the velocities and positions of boids in one cell were also contiguous in memory, so this data can be accessed directly and much more convenient than Uniform Grid neighbor search in part 2.1 .  The result is as I expected, since GPU performance will be better when dealing with continuous memory.
