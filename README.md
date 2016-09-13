**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Kaixiang Miao
* Tested on: Windows 7, i7-3630QM @ 2.40GHz 8GB, GTX 660M 2GB (Lenovo Y580 laptop, personal computer)

## Screenshot

___


![](./image/CoherentBoidsFlocking.gif)

>*Due to my use of gif tools [LICEcap](http://www.cockos.com/licecap/), my FPS gets much slower.*

The `.gif` above shows my work of **Coherent Boids Flocking**. Parameters are listed as below:

* *N\_FOR_VIS* : `10000`
* *DT* : `0.2f` 
* *blockSize* : `128`
* *rule1Dsitance* : `5.0f`
* *rule2Distance* : `3.0f`
* *rule3Distance* : `5.0f`
* *rule1Scale* : `0.01f`
* *rule2Scale* : `0.1f`
* *rule3Scale* : `0.1f`
* *maxSpeed* : `1.0f`
* *scene_scale* : `100.0f`
* *width * height* : `1280 * 720`
* *pointSize* : `2.0f`

## Performance Analysis

___

### Basic Analysis

The performance of **Naive Boids**, **Uniform Boids** and **Coherent Uniform Boids** is measured by FPS. Different amounts of boids are considered. The results are as below.
![](./image/table1.jpg)

![](./image/table2.jpg)

![](./image/table3.jpg)

In conclusion, the performance ranking is that **Coherent Uniform Boids > Uniform Boids > Naive Boids**. Besides, as the amount of boids increases, all the performance is weakened.

### Questions

* *For each implementation, how does changing the number of boids affect performance? Why do you think this is?*
	
As the number of boids increases, the performance in each implementation is slower. One of the insignificant reasons is that, in each implementation, the number of the GPU threads we use scales linearly with the amount of boids. This slightly increases the execution time and weakens our performance. However, it shouldn't be the major cause of the slow performance.

For **Naive Boids**, if not parallelized and assuming that *n* represents for the number of boids, the complexity of our algorithm for updating velocity of boids should be *O(n^2)*, since we have to iterate over each boid [ **step 1**] and for each boid, we have to iterate over other boids to calculate out the distance [ **step 2** ]. Luckily CUDA helps us reduce the complexity of the first part to *O(1)*. Hence the total complexity is *O(n)*.

For **Uniform Grid Boids**, optimization is done in the part of calculating out the distance. Being optimized by uniform grid using some additional indices buffer, the best complexity of this part is *O(1)*, in which each boid do not affect others. However, the worse complexity could be still *O(n)* considering the situation that all the boids are packed in one grid and affecting each other.

For **Coherent Uniform Grid Boids**, since it optimizes for the memory allocation but not the searching algorithm, the situation is the same as **Uniform Grid Boids**. What's more, `thrust::sort_by_key` should be taken into consideration in both of these methods.

* *For each implementation, how does changing the block count and block size affect performance? Why do you think this is?*

The result is quite interesting. I do the comparison as following:

![](./image/table4.jpg)

![](./image/table5.jpg)

![](./image/table6.jpg)
	
In order to achieve an obvious comparison, I push the amount of boids to the limit and keep FPS below 60. `boids = 10000` for **Naive Boids** and `boids = 100000` for both **Uniform Boids** and **Coherent Uniform Boids** fit very well.

As the charts show, changing the block count and block size does not obviously affect the performance while increasing the block size slowers the efficiency of **Uniform Boids** but improve the performance of **Coherent Uniform Boids**. To explain better, let's remind the steps I mentioned before.

> *... we have to iterate over each boid [ **step 1**] and for each boid, we have to iterate over other boids to calculate out the distance [ **step 2** ]*

The unaffected performance of **Naive Boids** has proved that [ **step 1**] won't be the crucial factor of this interesting result. Hence, the only difference between **Uniform Boids** and **Coherent Uniform Boids**, the memory allocation, should be the determinant. The pictures below may explain this.

![](./image/pic1.jpg)

![](./image/pic2.jpg)

If the data in the global memory is scattered, it will be tough for lots of threads in a large block to access them, since threads will be grouped into warps during the execution, as mentioned in [https://devblogs.nvidia.com/parallelforall/how-access-global-memory-efficiently-cuda-c-kernels/](https://devblogs.nvidia.com/parallelforall/how-access-global-memory-efficiently-cuda-c-kernels/). Rather, it's not efficient for lots of threads in many small blocks to access a large chunk of contiguous global memory.

* *For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?*
	
Yes. Actually, in the beginning I don't believe it will improve the performance since the rearrangement of the buffer seems more costly than the benefits gaining from the contiguous memory allocation. But later I found that the rearrangement can be totally parallelized and is quite efficient. The improvement of the performance gained by the contiguous memory is also beyond my expectation. The reason why the performance is improved by the more coherent uniform grid is that threads in the same block prefer to reading data which is stored in contiguous memory.
