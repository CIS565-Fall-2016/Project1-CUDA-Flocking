#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

#define INVALID_VALUE (-1)

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?



// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3* dev_sorted_pos;
glm::vec3* dev_sorted_vel1;
glm::vec3* dev_sorted_vel2;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

__global__ void kernCopyBuffer(int N, glm::vec3* desBuffer, glm::vec3* srcBuffer) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  desBuffer[index] = srcBuffer[index];
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  cudaMalloc((void**)&dev_sorted_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_sorted_pos failed!");
  
  cudaMalloc((void**)&dev_sorted_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_sorted_vel1 failed!");
  cudaMalloc((void**)&dev_sorted_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_sorted_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // Initialize the dev_sorted_pos
  kernCopyBuffer << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_sorted_pos, dev_pos);

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = halfSideCount * 2;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  dev_thrust_particleArrayIndices = thrust::device_pointer_cast<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_pointer_cast<int>(dev_particleGridIndices);

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");


  cudaThreadSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaThreadSynchronize();
}

/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  
  glm::vec3 velCohesion, velSeparation, velAlignment;
  glm::vec3 centerOfMass;
  unsigned int rule1NeighborCount = 0;
  unsigned int rule2NeighborCount = 0;
  unsigned int rule3NeighborCount = 0;
  
  for (auto i = 0; i < N; ++i) {
    
    // skip itself
    if (i == iSelf) continue;
    auto distance = glm::distance(pos[iSelf], pos[i]);

    // Rule 1, cohesion: boids fly towards their local perceived center of mass, which excludes themselves
    if (distance < rule1Distance) {
      centerOfMass += pos[i];
      ++rule1NeighborCount;
    }

    // Rule 2: boids try to stay a distance d away from each other
    if (distance < rule2Distance) {
      velSeparation -= (pos[i] - pos[iSelf]);
      ++rule2NeighborCount;
    }

    // Rule 3: boids try to match the speed of surrounding boids
    if (distance < rule3Distance) {
      velAlignment += vel[i];
      ++rule3NeighborCount;
    }
  }

  if (rule1NeighborCount > 0) {
    centerOfMass /= rule1NeighborCount;
    velCohesion = (centerOfMass - pos[iSelf]) * rule1Scale;
  }

  if (rule2NeighborCount > 0) {
    velSeparation *= rule2Scale;
  }

  if (rule3NeighborCount > 0) {
    velAlignment /= rule3NeighborCount;
    velAlignment = velAlignment - vel[iSelf];
    velAlignment *= rule3Scale;
  }

  // Return new computed velocity
  return velCohesion + velSeparation + velAlignment;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {

  // Compute a new velocity based on pos and vel1
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
    
  // Record the new velocity into vel2. Question: why NOT vel1?
  // Since the neighbor might be inspecting our current vel1 this frame, we don't want to update vel1 yet
  vel2[index] = vel1[index] + computeVelocityChange(N, index, pos, vel1);

  // Clamp the speed
  if (glm::length(vel2[index]) >= maxSpeed) {
    vel2[index] = glm::normalize(vel2[index]) * maxSpeed;
  }
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    indices[index] = index;

    // Compute the 3D grid indices for this particle
    int gridX = (int)((pos[index].x - gridMin.x) * inverseCellWidth);
    int gridY = (int)((pos[index].y - gridMin.y) * inverseCellWidth);
    int gridZ = (int)((pos[index].z - gridMin.z) * inverseCellWidth);

    // Convert and store the 1D grid index
    gridIndices[index] = gridIndex3Dto1D(gridX, gridY, gridZ, gridResolution);
  }
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index > N) {
    return;
  }

  int cellIndex = particleGridIndices[index];
  if (index == 0) {
    gridCellStartIndices[cellIndex] = 0;
    return;
  }

  if (index == N - 1) {
    gridCellEndIndices[cellIndex] = N - 1;
  }

  int prevCellIndex = particleGridIndices[index - 1];
  if (cellIndex != prevCellIndex)
  {
    gridCellStartIndices[cellIndex] = index;
    gridCellEndIndices[prevCellIndex] = index - 1;
  }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  // Compute a new velocity based on pos and vel1
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  int boidIndex = particleArrayIndices[index];

  // Compute the 3D grid indices for this particle
  glm::vec3 grid3DIndex = (pos[boidIndex] - gridMin) * inverseCellWidth;

    // Find the other 8 neiboring cells based on which octant the particle is in
  glm::vec3 distanceToNeighbor = 
    pos[boidIndex] - (glm::vec3((int)grid3DIndex.x, (int)grid3DIndex.y, (int)grid3DIndex.z) * cellWidth + gridMin);
  float halfCellWidth = cellWidth / 2.0f;

  int neighborMinIndexX, neighborMinIndexY, neighborMinIndexZ, neighborMaxIndexX, neighborMaxIndexY, neighborMaxIndexZ;

  if (distanceToNeighbor.x < halfCellWidth) {
    neighborMinIndexX = (int)grid3DIndex.x - 1 >= 0 ? (int)grid3DIndex.x - 1 : (int)grid3DIndex.x;
    neighborMaxIndexX = (int)grid3DIndex.x;
  } else {
    neighborMinIndexX = (int)grid3DIndex.x;
    neighborMaxIndexX = (int)grid3DIndex.x + 1 < N ? (int)grid3DIndex.x + 1: (int)grid3DIndex.x;
  }

  if (distanceToNeighbor.y < halfCellWidth) {
    neighborMinIndexY = (int)grid3DIndex.y - 1 >= 0 ? (int)grid3DIndex.y - 1 : (int)grid3DIndex.y;
    neighborMaxIndexY = (int)grid3DIndex.y;
  }
  else {
    neighborMinIndexY = (int)grid3DIndex.y;
    neighborMaxIndexY = (int)grid3DIndex.y + 1 < N ? (int)grid3DIndex.y + 1 : (int)grid3DIndex.y;
  }
  
  if (distanceToNeighbor.z < halfCellWidth) {
    neighborMinIndexZ = (int)grid3DIndex.z - 1 >= 0 ? (int)grid3DIndex.z - 1 : (int)grid3DIndex.z;
    neighborMaxIndexZ = (int)grid3DIndex.z;
  }
  else {
    neighborMinIndexZ = (int)grid3DIndex.z;
    neighborMaxIndexZ = (int)grid3DIndex.z + 1 < N ? (int)grid3DIndex.z + 1 : (int)grid3DIndex.z;
  }

  // These are to keep track of velocity computation
  glm::vec3 velCohesion, velSeparation, velAlignment;
  glm::vec3 centerOfMass;
  unsigned int rule1NeighborCount = 0;
  unsigned int rule2NeighborCount = 0;
  unsigned int rule3NeighborCount = 0;

  // Loop through neighbor cells and find all boids in them
  int neighborIndex, neighborCellStartIndex, neighborCellEndIndex;
  for (auto z = neighborMinIndexZ; z <= neighborMaxIndexZ; ++z) {
    for (auto y = neighborMinIndexY; y <= neighborMaxIndexY; ++y) {
      for (auto x = neighborMinIndexX; x <= neighborMaxIndexX; ++x) {
        
        // Compute velocity contribution from particles in this neighboring cell
        neighborIndex = gridIndex3Dto1D(x, y, z, gridResolution);
        neighborCellStartIndex = gridCellStartIndices[neighborIndex];
        neighborCellEndIndex = gridCellEndIndices[neighborIndex];

        if (neighborCellStartIndex >= 0 && neighborCellStartIndex <= neighborCellEndIndex &&
          neighborCellEndIndex >= neighborCellStartIndex && neighborCellEndIndex < N) {

          //Compute velocity
          for (auto i = neighborCellStartIndex; i < neighborCellEndIndex; ++i) {

            int neighborBoidIndex = particleArrayIndices[i];
            // skip itself
            if (neighborBoidIndex == boidIndex) continue;
            auto distance = glm::distance(pos[boidIndex], pos[neighborBoidIndex]);

            // Rule 1, cohesion: boids fly towards their local perceived center of mass, which excludes themselves
            if (distance < rule1Distance) {
              centerOfMass += pos[neighborBoidIndex];
              ++rule1NeighborCount;
            }

            // Rule 2: boids try to stay a distance d away from each other
            if (distance < rule2Distance) {
              velSeparation -= (pos[neighborBoidIndex] - pos[boidIndex]);
              ++rule2NeighborCount;
            }

            // Rule 3: boids try to match the speed of surrounding boids
            if (distance < rule3Distance) {
              velAlignment += vel1[neighborBoidIndex];
              ++rule3NeighborCount;
            }
          }          
        }
      }
    }
  }

  if (rule1NeighborCount > 0) {
    centerOfMass /= rule1NeighborCount;
    velCohesion = (centerOfMass - pos[boidIndex]) * rule1Scale;
  }

  if (rule2NeighborCount > 0) {
    velSeparation *= rule2Scale;
  }

  if (rule3NeighborCount > 0) {
    velAlignment /= rule3NeighborCount;
    velAlignment = velAlignment - vel1[boidIndex];
    velAlignment *= rule3Scale;
  }

  // Record the new velocity into vel2. Question: why NOT vel1?
  // Since the neighbor might be inspecting our current vel1 this frame, we don't want to update vel1 yet
  vel2[boidIndex] = vel1[boidIndex] + velCohesion + velSeparation + velAlignment;

  // Clamp the speed
  if (glm::length(vel2[boidIndex]) >= maxSpeed) {
    vel2[boidIndex] = glm::normalize(vel2[boidIndex]) * maxSpeed;
  }
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  // Compute the 3D grid indices for this particle
  glm::vec3 grid3DIndex = (pos[index] - gridMin) * inverseCellWidth;

  // Find the other 8 neiboring cells based on which octant the particle is in
  glm::vec3 distanceToNeighbor =
    pos[index] - (glm::vec3((int)grid3DIndex.x, (int)grid3DIndex.y, (int)grid3DIndex.z) * cellWidth + gridMin);
  float halfCellWidth = cellWidth / 2.0f;

  int neighborMinIndexX, 
    neighborMinIndexY, 
      neighborMinIndexZ, 
        neighborMaxIndexX, 
          neighborMaxIndexY, 
            neighborMaxIndexZ;

  if (distanceToNeighbor.x < halfCellWidth) {
    neighborMinIndexX = (int)grid3DIndex.x - 1;
    neighborMaxIndexX = (int)grid3DIndex.x;
  }
  else {
    neighborMinIndexX = (int)grid3DIndex.x;
    neighborMaxIndexX = (int)grid3DIndex.x + 1;
  }

  if (distanceToNeighbor.y < halfCellWidth) {
    neighborMinIndexY = (int)grid3DIndex.y - 1;
    neighborMaxIndexY = (int)grid3DIndex.y;
  }
  else {
    neighborMinIndexY = (int)grid3DIndex.y;
    neighborMaxIndexY = (int)grid3DIndex.y + 1;
  }

  if (distanceToNeighbor.z < halfCellWidth) {
    neighborMinIndexZ = (int)grid3DIndex.z - 1;
    neighborMaxIndexZ = (int)grid3DIndex.z;
  }
  else {
    neighborMinIndexZ = (int)grid3DIndex.z;
    neighborMaxIndexZ = (int)grid3DIndex.z + 1;
  }

  // These are to keep track of velocity computation
  glm::vec3 velCohesion, velSeparation, velAlignment;
  glm::vec3 centerOfMass;
  unsigned int rule1NeighborCount = 0;
  unsigned int rule2NeighborCount = 0;
  unsigned int rule3NeighborCount = 0;

  // Loop through neighbor cells and find all boids in them
  int neighborIndex, neighborCellStartIndex, neighborCellEndIndex;
  for (int z = neighborMinIndexZ; z <= neighborMaxIndexZ; ++z) {
    for (int y = neighborMinIndexY; y <= neighborMaxIndexY; ++y) {
      for (int x = neighborMinIndexX; x <= neighborMaxIndexX; ++x) {

        // Compute velocity contribution from particles in this neighboring cell
        neighborIndex = gridIndex3Dto1D(x % gridResolution, y % gridResolution, z % gridResolution, gridResolution);
        neighborCellStartIndex = gridCellStartIndices[neighborIndex];
        neighborCellEndIndex = gridCellEndIndices[neighborIndex];

        if (neighborCellStartIndex >= 0 && neighborCellStartIndex <= neighborCellEndIndex &&
          neighborCellEndIndex >= neighborCellStartIndex && neighborCellEndIndex < N) {


          //Compute velocity
          for (auto neighborBoidIndex = neighborCellStartIndex; neighborBoidIndex < neighborCellEndIndex; ++neighborBoidIndex) {

            // skip itself
            if (neighborBoidIndex == index) continue;
            auto distance = glm::distance(pos[index], pos[neighborBoidIndex]);

            // Rule 1, cohesion: boids fly towards their local perceived center of mass, which excludes themselves
            if (distance < rule1Distance) {
              centerOfMass += pos[neighborBoidIndex];
              ++rule1NeighborCount;
            }

            // Rule 2: boids try to stay a distance d away from each other
            if (distance < rule2Distance) {
              velSeparation -= (pos[neighborBoidIndex] - pos[index]);
              ++rule2NeighborCount;
            }

            // Rule 3: boids try to match the speed of surrounding boids
            if (distance < rule3Distance) {
              velAlignment += vel1[neighborBoidIndex];
              ++rule3NeighborCount;
            }
          }
        }
      }
    }
  }

  if (rule1NeighborCount > 0) {
    centerOfMass /= rule1NeighborCount;
    velCohesion = (centerOfMass - pos[index]) * rule1Scale;
  }

  if (rule2NeighborCount > 0) {
    velSeparation *= rule2Scale;
  }

  if (rule3NeighborCount > 0) {
    velAlignment /= rule3NeighborCount;
    velAlignment = velAlignment - vel1[index];
    velAlignment *= rule3Scale;
  }

  // Record the new velocity into vel2. Question: why NOT vel1?
  // Since the neighbor might be inspecting our current vel1 this frame, we don't want to update vel1 yet
  vel2[index] = vel1[index] + velCohesion + velSeparation + velAlignment;

  // Clamp the speed
  if (glm::length(vel2[index]) >= maxSpeed) {
    vel2[index] = glm::normalize(vel2[index]) * maxSpeed;
  } 
}

__global__ void kernRearrangeBuffer(int N, int* particleGridIndices, glm::vec3* newBuffer, glm::vec3* oldBuffer) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  int oldIdx = particleGridIndices[index];
  newBuffer[index] = oldBuffer[oldIdx];
}

__global__ void kernCopyBack(int N, int* particleGridIndices, glm::vec3* newBuffer, glm::vec3* oldBuffer) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  int newIdx = particleGridIndices[index];
  newBuffer[newIdx] = oldBuffer[index];
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  static bool isDevVel1Active = true;
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  
  kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, isDevVel1Active ? dev_vel1 : dev_vel2, isDevVel1Active ? dev_vel2 : dev_vel1);
  kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, isDevVel1Active ? dev_vel2 : dev_vel1);

  // TODO-1.2 ping-pong the velocity buffers
  isDevVel1Active = !isDevVel1Active;
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  // reseet the grid cell start and end indices to an invalid value
  kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> >(
    numObjects, 
      dev_gridCellStartIndices, 
      INVALID_VALUE
          );

  kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> >(
    numObjects, 
      dev_gridCellEndIndices, 
      INVALID_VALUE
          );

  // Compute the new index
  kernComputeIndices << <fullBlocksPerGrid, blockSize >> >(
    numObjects, 
      gridSideCount, 
        gridMinimum, 
          gridInverseCellWidth, 
          dev_pos,
              dev_particleArrayIndices, 
                dev_particleGridIndices
                  );

  thrust::sort_by_key(
    dev_thrust_particleGridIndices, 
      dev_thrust_particleGridIndices + numObjects, 
        dev_thrust_particleArrayIndices);

  kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> >(
    numObjects, 
      dev_particleGridIndices, 
        dev_gridCellStartIndices, 
          dev_gridCellEndIndices
            );

  // Update velocity
  static bool isDevVel1Active = true;
  kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> >(
    numObjects,
      gridSideCount, 
        gridMinimum, 
          gridInverseCellWidth, 
            gridCellWidth, 
              dev_gridCellStartIndices, 
                dev_gridCellEndIndices, 
                  dev_particleArrayIndices, 
                    dev_pos, 
                      isDevVel1Active ? dev_vel1 : dev_vel2, 
                        isDevVel1Active ? dev_vel2 : dev_vel1
                          );
  
  //  Update position
  kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(
    numObjects, 
      dt, 
        dev_pos, 
          isDevVel1Active ? dev_vel2 : dev_vel1
            );
  
  // Ping pong
  isDevVel1Active = !isDevVel1Active;
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.

  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  // reseet the grid cell start and end indices to an invalid value
  kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> >(
    numObjects,
    dev_gridCellStartIndices,
    INVALID_VALUE
    );

  kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> >(
    numObjects,
    dev_gridCellEndIndices,
    INVALID_VALUE
    );

  // Compute the new index
  kernComputeIndices << <fullBlocksPerGrid, blockSize >> >(
    numObjects,
    gridSideCount,
    gridMinimum,
    gridInverseCellWidth,
    dev_pos,
    dev_particleArrayIndices,
    dev_particleGridIndices
    );

  thrust::sort_by_key(
    dev_thrust_particleGridIndices,
    dev_thrust_particleGridIndices + numObjects,
    dev_thrust_particleArrayIndices);

  kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> >(
    numObjects,
    dev_particleGridIndices,
    dev_gridCellStartIndices,
    dev_gridCellEndIndices
    );

  kernRearrangeBuffer << <fullBlocksPerGrid, blockSize >> >(
    numObjects,
    dev_particleArrayIndices,
    dev_sorted_pos,
    dev_pos
    );

  kernRearrangeBuffer << <fullBlocksPerGrid, blockSize >> >(
    numObjects,
    dev_particleArrayIndices,
    dev_vel1,
    dev_vel2
    );

  // Update velocity
  kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> >(
    numObjects,
    gridSideCount,
    gridMinimum,
    gridInverseCellWidth,
    gridCellWidth,
    dev_gridCellStartIndices,
    dev_gridCellEndIndices,
    dev_sorted_pos,
    dev_vel1,
    dev_vel2
    );

    //  Update position
  kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(
    numObjects,
    dt,
    dev_sorted_pos,
    dev_vel2
    );

  // Ping pong
  std::swap(dev_pos, dev_sorted_pos);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_sorted_pos);
  cudaFree(dev_sorted_vel1);
  cudaFree(dev_sorted_vel2);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  int *intKeys = new int[N];
  int *intValues = new int[N];

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues, sizeof(int) * N, cudaMemcpyHostToDevice);

  // ------------

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys, dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues, dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // ------------
  float test_cellWidth = 10.0;
  float test_sceneScale = 100.0f;
  int test_sideCount = (int)(test_sceneScale / test_cellWidth);
  int test_halfSideCount = test_sideCount / 2;
  int test_gridCellCount = test_sideCount * test_sideCount * test_sideCount;
  float test_inverseCellWidth = 1.0f / test_cellWidth;
  float test_halfGridWidth = test_cellWidth * test_halfSideCount;
  glm::vec3 test_gridMinimum(-test_halfGridWidth, -test_halfGridWidth, -test_halfGridWidth);

  std::cout << "Test cell width: " << test_cellWidth << std::endl;
  std::cout << "Test scene scale: " << test_sceneScale << std::endl;
  std::cout << "Test half side count: " << test_halfSideCount << std::endl;
  std::cout << "Test side count: " << test_sideCount << std::endl;
  std::cout << "Test grid cell count: " << test_gridCellCount << std::endl;
  std::cout << "Test inverse cell width: " << test_inverseCellWidth << std::endl;
  std::cout << "Test half grid width: " << test_halfGridWidth << std::endl;
  std::cout << "Test grid minimum: " << test_gridMinimum.x << ", " << test_gridMinimum.y << ", " << test_gridMinimum.z << std::endl;

  int* test_particleArrayIndices = new int[N];
  int* test_particleGridIndices = new int[N];
  glm::vec3* test_pos = new glm::vec3[N];

  glm::vec3* test_dev_pos;
  int* test_dev_particleArrayIndices;
  int* test_dev_particleGridIndices;

  cudaMalloc((void**)&test_dev_pos, N * sizeof(glm::vec3));
  for (int i = 0; i < N; ++i) {
    test_pos[i] = glm::vec3(test_cellWidth * i) + test_gridMinimum;

  }
  cudaMemcpy(test_dev_pos, test_pos, sizeof(glm::vec3) * N, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&test_dev_particleArrayIndices, N * sizeof(int));
  cudaMalloc((void**)&test_dev_particleGridIndices, N * sizeof(int));

  kernComputeIndices << <1, N >> >(
    N, 
      test_sideCount,
        test_gridMinimum,
          test_inverseCellWidth,
            test_dev_pos,
              test_dev_particleArrayIndices,
                test_dev_particleGridIndices
                  );

  cudaMemcpy(test_pos, test_dev_pos, sizeof(glm::vec3) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(test_particleArrayIndices, test_dev_particleArrayIndices, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(test_particleGridIndices, test_dev_particleGridIndices, sizeof(int) * N, cudaMemcpyDeviceToHost);

  std::cout << "Test kernComputeIndices  : " << std::endl;
  for (auto i = 0; i < N; i++) {
    std::cout << "  pos: " << test_pos[i].x << ", " << test_pos[i].y << ", " << test_pos[i].z;
    std::cout << "  particle array indices: " << test_particleArrayIndices[i];
    std::cout << " particle grid indices: " << test_particleGridIndices[i] << std::endl;
  }

  // ------------

  int *dev_keyStarts;
  int *dev_keyEnds;
  cudaMalloc((void**)&dev_keyStarts, N * sizeof(int));
  cudaMalloc((void**)&dev_keyEnds, N * sizeof(int));

  int *intStarts = new int[N];
  int *intEnds = new int[N];

  kernResetIntBuffer << <1, N >> >(N, dev_keyStarts, INVALID_VALUE);
  kernResetIntBuffer << <1, N >> >(N, dev_keyEnds, INVALID_VALUE);
  cudaMemcpy(intStarts, dev_keyStarts, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intEnds, dev_keyEnds, sizeof(int) * N, cudaMemcpyDeviceToHost);

  std::cout << "Test kernResetIntBuffer  : " << std::endl;
  for (auto i = 0; i < N; i++) {
    std::cout << "  start: " << intStarts[i];
    std::cout << " end: " << intEnds[i] << std::endl;
  }

  // ------------

  kernIdentifyCellStartEnd << <1, N >> >(N, dev_intKeys, dev_keyStarts, dev_keyEnds);

  cudaMemcpy(intStarts, dev_keyStarts, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intEnds, dev_keyEnds, sizeof(int) * N, cudaMemcpyDeviceToHost);

  std::cout << "Test kernIdentifyCellStartEnd  : " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  start: " << intStarts[i];
    std::cout << " end: " << intEnds[i] << std::endl;
  }

  // --------------
  glm::vec3* test_sorted_pos = new glm::vec3[N];

  glm::vec3* test_dev_sorted_pos;
  cudaMalloc((void**)&test_dev_sorted_pos, N * sizeof(glm::vec3));

  std::cout << "Test kernRearrangeBuffer  : " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  pos: " << test_pos[i].x << ", " << test_pos[i].y << ", " << test_pos[i].z;
    std::cout << " index: " << intValues[i] << std::endl;
  }
  kernRearrangeBuffer << <1, N >> >(N, dev_intValues, test_dev_sorted_pos, test_dev_pos);
  cudaMemcpy(test_sorted_pos, test_dev_sorted_pos, sizeof(glm::vec3) * N, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    std::cout << "  sorted_pos: " << test_sorted_pos[i].x << ", " << test_sorted_pos[i].y << ", " << test_sorted_pos[i].z;
    std::cout << " index: " << intValues[i] << std::endl;
  }


  // cleanup
  delete[] intKeys;
  delete[] intValues;
  delete[] intStarts;
  delete[] intEnds;
  delete[] test_particleArrayIndices;
  delete[] test_particleGridIndices;
  delete[] test_pos;
  delete[] test_sorted_pos;
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  cudaFree(dev_keyStarts);
  cudaFree(dev_keyEnds);
  cudaFree(test_dev_pos);
  cudaFree(test_dev_particleArrayIndices);
  cudaFree(test_dev_particleGridIndices);
  cudaFree(test_dev_sorted_pos);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
