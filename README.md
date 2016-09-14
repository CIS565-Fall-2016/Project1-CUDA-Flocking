# **University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Ruoyu Fan
* Tested on:  Windows 7, Xeon(R) E5-1630 @ 3.70GHz 32GB, GTX 1070 8192MB (Moore 103 SigLab)

![simulation preview](/screenshots/flocking.gif)

### Description - CUDA Flocking

I implemented flocking simulation (Boids) using three different neighbor searching approaches: 
 * **Brute-force**
 * **Scattered grid**
 * **Coherent grid**

#### Q&A
1. For each implementation, how does changing the number of boids affect performance? Why do you think this is?
    **Ans:** _Working on it._
2. For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
    **Ans:** _Working on it._
3. For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
    **Ans:** _Yes. It is. because coherent approach follows the principle of "keeping things you will use close" and causes the pointers jump around less._

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)
