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
 
#### Performance 
| Particle Count|5000|10000|20000|40000|80k|160k|320k|640k|1280k
| ------------- |-----:| -----:|
| Brute-force FPS|591.5|242.1|52.5|14.5|4.2
| Scattered Grid FPS|||1103.8|1072|662.1|344.7|121.5|37.2|10.9
| Coherent Grid  FPS|||1069.1|1082.4|761.8|468.9|191.3|63.5|17.9

#### Q&A
1. For each implementation, how does changing the number of boids affect performance? Why do you think this is?
    
	**Ans:** _Working on it._

2. For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
    
	**Ans:** _Working on it._

3. For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
    
	**Ans:** _Yes. It is. because coherent approach follows the principle of "keeping things you will use close" and causes the pointers jump around less._

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)
