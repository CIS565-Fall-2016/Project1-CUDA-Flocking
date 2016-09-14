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

	**Ans:** _Yes. It is. because coherent approach follows the principle of "keeping things you will use close" and causes the pointers jump around less. So it is faster when using a uniform grid and access the memory continuously._


#### Performance
| Particle Count|5000|10000|20000|40000|80k|160k|320k|640k|1280k
| ------------- |-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| Brute-force FPS|591.5|242.1|52.5|14.5|4.2|||||
| Scattered Grid FPS|||1103.8|1072|662.1|344.7|121.5|37.2|10.9
| Coherent Grid  FPS|||1069.1|1082.4|761.8|468.9|191.3|63.5|17.9

Data for the performance impact of block size was left on the lab machine... Will retrieve it later.
