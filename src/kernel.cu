#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

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
#define blockSize 1024

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
glm::vec3 *dev_pos1;
glm::vec3 *dev_pos2;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos1 and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

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

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos1 failed!");

  cudaMalloc((void**)&dev_pos2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos2 failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos1, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  glm::vec3 *zero = new glm::vec3[N];
  cudaMemcpy(dev_vel1, zero, N*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_vel2, zero, N*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  delete zero;

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N*sizeof(int));
  cudaMalloc((void**)&dev_particleGridIndices, N*sizeof(int));
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount*sizeof(int));
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount*sizeof(int));
  cudaThreadSynchronize();

  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

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

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos1, vbodptr_positions, scene_scale);
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
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids

  glm::vec3 ctr(0.0);
  glm::vec3 v1(0.0), v2(0.0), v3(0.0);
  float k1=0.0, k2=0.0, k3=0.0;
  for (int i = 0; i < N; i++) {
    if (i == iSelf)
      continue;

    float dist = glm::length(pos[i] - pos[iSelf]);

    if (dist < rule1Distance) {
      ctr += pos[i];
      k1++;
    }

    if (dist < rule2Distance) {
      v2 += pos[iSelf] - pos[i];
      k2++;
    }

    if (dist < rule3Distance) {
      v3 += vel[i];
      k3++;
    }
  }

  glm::vec3 dVel(0.0);
  if (k1 > 0)
    dVel += rule1Scale * (ctr/k1 - pos[iSelf]);
  if (k2 > 0)
    dVel += rule2Scale * v2;
  if (k3 > 0)
    dVel += rule3Scale * (v3/k3 - vel[iSelf]);

  return dVel;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?

  // get the self index
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N)
    return;

  glm::vec3 dVel = computeVelocityChange(N, index, pos, vel1);
  vel2[index] += dVel;
  float speed = glm::length(vel2[index]);
  if (speed > maxSpeed)
    vel2[index] *= maxSpeed / speed;

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

  // this is run before sorting

  int boidIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (boidIdx >= N)
    return;

  glm::ivec3 gridIdx3 = (pos[boidIdx]-gridMin)*inverseCellWidth;
  int gridIdx = gridIndex3Dto1D(gridIdx3.x, gridIdx3.y, gridIdx3.z,
                                gridResolution);
  gridIndices[boidIdx] = gridIdx;
  indices[boidIdx] = boidIdx;
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

  // this is post-sorting

  int boidIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (boidIdx >= N)
    return;

  int boidCell = particleGridIndices[boidIdx];

  // short circuited, adjacent index checks only happen if not at
  // array boundary

  if (boidIdx == 0 || boidCell != particleGridIndices[boidIdx-1]) {
    gridCellStartIndices[boidCell] = boidIdx;
  }

  if (boidIdx == N-1 || boidCell != particleGridIndices[boidIdx+1]) {
    gridCellEndIndices[boidCell] = boidIdx;
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

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;
  int boidIdx = particleArrayIndices[idx];

  // get the grid index and offset
  glm::vec3 fGridIdx3;
  glm::vec3 relPos = glm::modf((pos[boidIdx]-gridMin) * inverseCellWidth, fGridIdx3);
  relPos = cellWidth * relPos + gridMin / float(gridResolution);
  glm::ivec3 gridIdx3 = (glm::ivec3)fGridIdx3;
  int gridIdx = gridIndex3Dto1D(gridIdx3.x, gridIdx3.y, gridIdx3.z,
                              gridResolution);

  // find which adjacent cells to search
  glm::ivec3 searchIdx(0,0,0);
  for (int i = 0; i < 3; i++) {
    if (gridIdx3[i] > 0 && relPos[i] < 0)
      searchIdx[i] = -1;
    if (gridIdx3[i] < gridResolution-1 && relPos[i] > 0)
      searchIdx[i] = 1;
  }


  // search all extant adjacent cells
  glm::ivec3 sMin = glm::min(gridIdx3 + searchIdx, gridIdx3);
  glm::ivec3 sMax = glm::max(gridIdx3 + searchIdx, gridIdx3);
  glm::vec3 ctr(0.0);
  glm::vec3 v1(0.0), v2(0.0), v3(0.0);
  float k1=0.0, k2=0.0, k3=0.0;
  for (int k = sMin.z; k <= sMax.z; k++) {
  for (int j = sMin.y; j <= sMax.y; j++) {
  for (int i = sMin.x; i <= sMax.x; i++) {
    gridIdx = gridIndex3Dto1D(i,j,k,gridResolution);
    int cellStart = gridCellStartIndices[gridIdx];
    int cellEnd = gridCellEndIndices[gridIdx];
    if (cellStart < 0 || cellEnd < 0)
      continue;

    for (int p = cellStart; p <= cellEnd; p++) {
      int pBoid = particleArrayIndices[p];
      if (pBoid == boidIdx)
        continue;

      float dist = glm::length(pos[pBoid] - pos[boidIdx]);

      if (dist < rule1Distance) {
        ctr += pos[pBoid];
        k1++;
      }

      if (dist < rule2Distance) {
        v2 += pos[boidIdx] - pos[pBoid];
        k2++;
      }

      if (dist < rule3Distance) {
        v3 += vel1[pBoid];
        k3++;
      }
    }
  }}}

  // total velocity change
  glm::vec3 nVel = vel1[boidIdx];
  if (k1 > 0)
    nVel += rule1Scale * (ctr/k1 - pos[boidIdx]);
  if (k2 > 0)
    nVel += rule2Scale * v2;
  if (k3 > 0)
    nVel += rule3Scale * (v3/k3 - vel1[boidIdx]);

  float speed = glm::length(nVel);
  if (speed > maxSpeed)
    nVel *= maxSpeed / speed;
  vel2[boidIdx] = nVel;
}

// shuffle data1 into data2 according to a given order
__global__ void kernShuffleToOrder(int N, glm::vec3 *data1, glm::vec3 *data2, int *order) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;
  int src = order[idx];

  data2[idx] = data1[src];
}

__global__ void kernUpdateVelNeighborSearchCoherent(
    int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int *gridCellStartIndices, int *gridCellEndIndices,
    glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  int boidIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (boidIdx >= N)
    return;

  // get the grid index and offset
  glm::vec3 fGridIdx3;
  glm::vec3 relPos = glm::modf((pos[boidIdx]-gridMin) * inverseCellWidth, fGridIdx3);
  relPos = cellWidth * relPos + gridMin / float(gridResolution);
  glm::ivec3 gridIdx3 = (glm::ivec3)fGridIdx3;
  int gridIdx = gridIndex3Dto1D(gridIdx3.x, gridIdx3.y, gridIdx3.z,
                              gridResolution);

  // find which adjacent cells to search
  glm::ivec3 searchIdx(0,0,0);
  for (int i = 0; i < 3; i++) {
    if (gridIdx3[i] > 0 && relPos[i] < 0)
      searchIdx[i] = -1;
    if (gridIdx3[i] < gridResolution-1 && relPos[i] > 0)
      searchIdx[i] = 1;
  }


  // search all extant adjacent cells
  glm::ivec3 sMin = glm::min(gridIdx3 + searchIdx, gridIdx3);
  glm::ivec3 sMax = glm::max(gridIdx3 + searchIdx, gridIdx3);
  glm::vec3 ctr(0.0);
  glm::vec3 v1(0.0), v2(0.0), v3(0.0);
  float k1=0.0, k2=0.0, k3=0.0;
  for (int k = sMin.z; k <= sMax.z; k++) {
  for (int j = sMin.y; j <= sMax.y; j++) {
  for (int i = sMin.x; i <= sMax.x; i++) {
    gridIdx = gridIndex3Dto1D(i,j,k,gridResolution);
    int cellStart = gridCellStartIndices[gridIdx];
    int cellEnd = gridCellEndIndices[gridIdx];
    if (cellStart < 0 || cellEnd < 0)
      continue;

    for (int pBoid = cellStart; pBoid <= cellEnd; pBoid++) {
      if (pBoid == boidIdx)
        continue;

      float dist = glm::length(pos[pBoid] - pos[boidIdx]);

      if (dist < rule1Distance) {
        ctr += pos[pBoid];
        k1++;
      }

      if (dist < rule2Distance) {
        v2 += pos[boidIdx] - pos[pBoid];
        k2++;
      }

      if (dist < rule3Distance) {
        v3 += vel1[pBoid];
        k3++;
      }
    }
  }}}

  // total velocity change
  glm::vec3 nVel = vel1[boidIdx];
  if (k1 > 0)
    nVel += rule1Scale * (ctr/k1 - pos[boidIdx]);
  if (k2 > 0)
    nVel += rule2Scale * v2;
  if (k3 > 0)
    nVel += rule3Scale * (v3/k3 - vel1[boidIdx]);

  float speed = glm::length(nVel);
  if (speed > maxSpeed)
    nVel *= maxSpeed / speed;
  vel2[boidIdx] = nVel;
}


/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
// TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
// TODO-1.2 ping-pong the velocity buffers
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  //std::cout << "naive step" << std::endl;

  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects,
      dev_pos1, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

  cudaMemcpy(dev_vel1, dev_vel2, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos1, dev_vel1);
  checkCUDAErrorWithLine("kernUpdatePos failed!");
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

  //std::cout << "scatteredGrid step" << std::endl;

  dim3 boid_fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 cell_fullBlocksPerGrid((gridCellCount + blockSize - 1) / blockSize);

  // initialize indices
  kernComputeIndices<<<boid_fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount,
      gridMinimum, gridInverseCellWidth, dev_pos1,
      dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  // unstable sort
  thrust::sort_by_key(dev_thrust_particleGridIndices,
      dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // initialize all grid cells to 'unoccupied'
  kernResetIntBuffer<<<cell_fullBlocksPerGrid, blockSize>>>(gridCellCount,
      dev_gridCellStartIndices, -7);
  kernResetIntBuffer<<<cell_fullBlocksPerGrid, blockSize>>>(gridCellCount,
      dev_gridCellEndIndices, -7);

  // find cell boundaries
  kernIdentifyCellStartEnd<<<boid_fullBlocksPerGrid, blockSize>>>(numObjects,
      dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  // perform flocking rules
  kernUpdateVelNeighborSearchScattered<<<boid_fullBlocksPerGrid, blockSize>>>(
    numObjects, gridSideCount, gridMinimum,
    gridInverseCellWidth, gridCellWidth,
    dev_gridCellStartIndices, dev_gridCellEndIndices,
    dev_particleArrayIndices,
    dev_pos1, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

  // update positions
  kernUpdatePos<<<boid_fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos1,
      dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // ping-pong
  cudaMemcpy(dev_vel1, dev_vel2, numObjects*sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
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

  dim3 boid_fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 cell_fullBlocksPerGrid((gridCellCount + blockSize - 1) / blockSize);

  // initialize indices
  kernComputeIndices<<<boid_fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount,
      gridMinimum, gridInverseCellWidth, dev_pos1,
      dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  // unstable sort
  thrust::sort_by_key(dev_thrust_particleGridIndices,
      dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // initialize all grid cells to 'unoccupied'
  kernResetIntBuffer<<<cell_fullBlocksPerGrid, blockSize>>>(gridCellCount,
      dev_gridCellStartIndices, -7);
  kernResetIntBuffer<<<cell_fullBlocksPerGrid, blockSize>>>(gridCellCount,
      dev_gridCellEndIndices, -7);

  // find cell boundaries
  kernIdentifyCellStartEnd<<<boid_fullBlocksPerGrid, blockSize>>>(numObjects,
      dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  // reorder pos and vel
  kernShuffleToOrder<<<boid_fullBlocksPerGrid, blockSize>>>(numObjects,
      dev_pos1, dev_pos2, dev_particleArrayIndices);
  checkCUDAErrorWithLine("kernShuffleToOrder failed!");
  kernShuffleToOrder<<<boid_fullBlocksPerGrid, blockSize>>>(numObjects,
      dev_vel1, dev_vel2, dev_particleArrayIndices);
  checkCUDAErrorWithLine("kernShuffleToOrder failed!");
  /*
  thrust::gather(dev_thrust_particleArrayIndices,
      dev_thrust_particleArrayIndices+numObjects, thrust_pos1, thrust_pos2);
  thrust::gather(dev_thrust_particleArrayIndices,
      dev_thrust_particleArrayIndices+numObjects, thrust_vel1, thrust_vel2);
  */

  // perform flocking rules
  kernUpdateVelNeighborSearchCoherent<<<boid_fullBlocksPerGrid, blockSize>>>(
    numObjects, gridSideCount, gridMinimum,
    gridInverseCellWidth, gridCellWidth,
    dev_gridCellStartIndices, dev_gridCellEndIndices,
    dev_pos2, dev_vel2, dev_vel1);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");

  // ping-pong
  cudaMemcpy(dev_pos1, dev_pos2, numObjects*sizeof(glm::vec3),
      cudaMemcpyDeviceToDevice);

  // update positions
  kernUpdatePos<<<boid_fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos1,
      dev_vel1);
  checkCUDAErrorWithLine("kernUpdatePos failed!");
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos1);
  cudaFree(dev_pos2);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
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

  // cleanup
  delete[] intKeys;
  delete[] intValues;
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
