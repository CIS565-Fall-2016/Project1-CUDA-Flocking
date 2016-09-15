# **University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Ruoyu Fan
* Tested on:  Windows 7, Xeon(R) E5-1630 @ 3.70GHz 32GB, GTX 1070 8192MB (Moore 103 SigLab)

### Description - CUDA Flocking

I implemented flocking simulation (Boids) using three different neighbor searching approaches:
 * **Brute-force**
 * **Scattered grid**
 * **Coherent grid**

 ![simulation preview](/screenshots/flocking.gif)

#### Q&A
1. For each implementation, how does changing the number of boids affect performance? Why do you think this is?

	**Ans:** _In brute-force method, the framerate drops significantly faster than the other two methods. The framerate drops at square rate when the number of boids (threads) reaches the concurrency limit of GPU._

    _In the other two optimized methods, as the number of boids increase, the framerate drops much slower than the naive method._

	
2. For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

	**Ans:** _If the block is very small, the performance will drop very fast. In my opinion the first reason is that if the block size is too small (maybe less than warp size), some of the processing units are wasted. Increasing block size can improve performance, but the effect will be less obvious when the block is large enough - and it is limited by hardware capacity. Large block size can also cause some threads to be wasted._

	
3. For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

	**Ans:** _Yes. It is. because coherent approach follows the principle of "keeping things you will use close" and results in less pointer jumping and less page access. So it is faster when using a uniform grid and access the memory continuously._


#### Performance

On grid size of 128 and visual disabled, other settings as source code. 

| Particle Count|Naive FPS|Scattered Grid FPS|Coherent Grid FPS
| ------------- |--------:|-----------------:|----------------:|
| 5000|591.5||
| 10000|242.1||
| 20000|52.5|1103.8|1069.1
| 40000|14.5|1072.0|1082.4
| 80k|4.2|662.1|761.8
| 160k||344.7|468.9
| 320k||121.5|191.3
| 640k||37.2|63.5
| 1280k||10.9|17.9


On 160000 particles and visual disabled, other settings as source code.

| Particle Count|Scattered Grid FPS|Coherent Grid FPS
| --------------|-----------------:|----------------:|
| 4|115.5|154.4
| 8|185.5|245.7
| 16|268.8|259.9
| 32|351.2|465.8
| 64|349.2|463.3
| 128|347.8|472.3
| 256|348.1|464.4
| 512|347.9|465.7
| 1024|343.5|467.3

