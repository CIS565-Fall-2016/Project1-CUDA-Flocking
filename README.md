#University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1 - Flocking#

## Xiaomao Ding ##
* Tested on: Windows 8.1, i7-4700MQ @ 2.40GHz 8.00GB, GT 750M 2047MB (Personal Computer)

<div style="text-align:center"><img src ="https://github.com/xnieamo/Project1-CUDA-Flocking/blob/master/images/dt0.2_particles16000.gif" /></div>

Above is a gif generated using the code in this repo with 16000 boids using the coherent grid implementation.

### Quick Note ###
Before running any of the code in this repo, it is possible that you may have to adjust the compute capability flag in `scr/CMakeLists.txt`. To do so, change the '-arch=sm_30' to match your compute capability. 20 matches to 2.0, 30 to 3.0, etc.

## Performance Analysis ##

