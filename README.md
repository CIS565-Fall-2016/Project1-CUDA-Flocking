**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Xiang Deng
* Tested on: Windows 7, i7-3610QM @ 2.3GHz 8GB, GTX 660M 1024MB 

### DEMO

* With 5000 particles:
![](images/5000.gif)

* With 50000 particles:
![](images/50000.gif)

### Performance Analysis
The visualization has been disabled so that the performance analysis is based on the CUDA simulation only.
* Performance comparison with blocksize fixed (128).

![](images/performance-1.JPG)

|                        | 5000 particles | 7500 particles | 10000 particles | 15000 particles | 20000 particles |30000 particles|
|------------------------|----------------|-----------------|-----------------|------------------|------------------|------------------|
| Naive search (avg cuda time per frame (ms))           |    0.0185    |    0.0388    |    0.0633      |    0.122    |0.188 | 0.28|
| Scattered uniform grid (avg cuda time per frame (ms))  |    0.00192  |      0.00269 |     0.00355    |       0.0067     |0.00752 |0.0121 |
| Coherent uniform grid (avg cuda time per frame (ms))  |    0.00126  |   0.0014   |    0.00169   |    0.00232        |0.00278 | 0.00407|
* Performance comparison with particle number fixed (5000).

![](images/performance-2.JPG)

|                        | Blocksize=64| Blocksize=128 | Blocksize=192| Blocksize=256 |  
|------------------------|----------------|-----------------|-----------------|------------------| 
| Naive search (avg cuda time per frame (ms))           |    0.02012    |    0.01861   |   0.02178     |  0.01863 |
| Scattered uniform grid (avg cuda time per frame (ms))  |    0.00197  |     0.0022 |     0.0022  | 0.00191  |
| Coherent uniform grid (avg cuda time per frame (ms))  |    0.00119  |   0.00115  |   0.00118   |    0.00125       | 

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)
