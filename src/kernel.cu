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
glm::vec3 *dev_shuffledPos;
glm::vec3 *dev_shuffledVel;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;				// number of cells in the scene = gridSideCount^3
int gridSideCount;				// number of cells along one side
float gridCellWidth;			// width of cell
float gridInverseCellWidth;		// inverse of gridCellWidth
glm::vec3 gridMinimum;			// minimum grid coordinates

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
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray << <fullBlocksPerGrid, blockSize >> >(1, numObjects,
	  dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

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
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!"); 
  
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);

  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  cudaMalloc((void**)&dev_shuffledPos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_shuffledPos failed!");

  cudaMalloc((void**)&dev_shuffledVel, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_shuffledVel failed!");

  //cudaThreadSynchronize(); deprecated!
  cudaDeviceSynchronize();
  
  // init bActiveDevice for ping-pong buffer
  bActiveDevice = false;
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

/*
* three Boids velocity rules according to http://www.vergenet.net/~conrad/boids/pseudocode.html
* rule1 - mass center
* rule2 - seperate
* rule3 - cohesion
*/
__device__ glm::vec3 getVelMassCenter(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel)
{
	glm::vec3 velCenter(0, 0, 0);
	int neighborCount = 0;

	for (int i = 0; i < N; ++i)
	{
		if (i != iSelf && glm::length(pos[i] - pos[iSelf]) < rule1Distance)
		{
			velCenter += pos[i];
			neighborCount++;
		}
	}

	if (neighborCount > 0)
	{
		velCenter = velCenter / float(neighborCount);
		return (velCenter - pos[iSelf]) * rule1Scale;
	}
	else
	{
		return velCenter;
	}

}

__device__ glm::vec3 getVelSeperation(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel)
{
	glm::vec3 velSep(0, 0, 0);
	//int neighborCount = 0;

	for (int i = 0; i < N; ++i)
	{
		if (i != iSelf && glm::length(pos[i] - pos[iSelf]) < rule2Distance)
		{
			velSep = velSep - (pos[i] - pos[iSelf]);
		//	neighborCount++;
		}
	}

	return velSep * rule2Scale;
}

__device__ glm::vec3 getVelCohesion(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel)
{
	glm::vec3 velCohesion(0, 0, 0);
	int neighborCount = 0;

	for (int i = 0; i < N; ++i)
	{
		if (i != iSelf && glm::length(pos[i] - pos[iSelf]) < rule3Distance)
		{
			velCohesion += vel[i];
			neighborCount++;
		}
	}

	if (neighborCount > 0)
	{
		velCohesion = velCohesion / float(neighborCount);
		return (velCohesion - vel[iSelf]) * rule3Scale;
	}
	else
	{
		return velCohesion;
	}
}


__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {

  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	glm::vec3 velCenter = getVelMassCenter(N, iSelf, pos, vel);

  // Rule 2: boids try to stay a distance d away from each other
	glm::vec3 velSeperation = getVelSeperation(N, iSelf, pos, vel);

  // Rule 3: boids try to match the speed of surrounding boids
	glm::vec3 velCohesion = getVelCohesion(N, iSelf, pos, vel);

	// combine all rules, return new Velocity
	return velCenter + velSeperation + velCohesion;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {

	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}

  // Compute a new velocity based on pos and vel1
	glm::vec3 newVel = vel1[index] + computeVelocityChange(N, index, pos, vel1);

  // Clamp the speed
	if (glm::length(newVel) > maxSpeed)
	{
		newVel = glm::normalize(newVel) * maxSpeed;
	}

  // Record the new velocity into vel2. Question: why NOT vel1?
	vel2[index] = newVel;
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
	if (index >= N)
		return;

	// label its index (particleArrayIndices)
	indices[index] = index; 

	// label its grid cell index (particleGridIndices)
	glm::vec3 diff = pos[index] - gridMin;
	diff *= inverseCellWidth;
	int gridIndex = gridIndex3Dto1D(int(diff.x), int(diff.y), int(diff.z), gridResolution);

	gridIndices[index] = gridIndex;

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

	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
		return;

	int previous = index - 1;
	int next = index + 1;
	int gridIdx = particleGridIndices[index];

	// first cell
	if (previous < 0)
	{
		gridCellStartIndices[gridIdx] = index;
		return;
	}
	// last cell
	if (next >= N)
	{
		gridCellEndIndices[gridIdx] = index;

		//!!! NO return !!!///
		//return;
	}

	// new cell when previous index is different with current index
	int preGridIdx = particleGridIndices[previous];
	if (preGridIdx != gridIdx)
	{
		gridCellEndIndices[preGridIdx] = previous;
		gridCellStartIndices[gridIdx] = index;
	}
}


/*
* helper function for calculating all particles within a grid given girdCellIndex
*/
__device__ void computeVelocitiesWithinGivenGrid(
	int gridIndex, int particleIndex,
	int* gridCellStartIndices, int* gridCellEndIndices, int* particleArrayIndices,
	glm::vec3* pos, glm::vec3* vel,
	glm::vec3* velocities, int* neighborCounts)
{

	if (gridCellStartIndices[gridIndex] != -1 && gridCellEndIndices[gridIndex] != -1)
	{
		for (int i = gridCellStartIndices[gridIndex]; i <= gridCellEndIndices[gridIndex]; ++i)
		{
			int thisParticleIndex = particleArrayIndices[i];
			if (thisParticleIndex == particleIndex)
				continue;

			glm::vec3 thisParticlePos = pos[thisParticleIndex];
			float distance = glm::length(thisParticlePos - pos[particleIndex]);

			// rule1: center of mass
			if (distance <= rule1Distance)
			{
				velocities[0] += thisParticlePos;
				neighborCounts[0]++;
			}

			// rule2: separation
			if (distance <= rule2Distance)
			{
				velocities[1] = velocities[1] - (thisParticlePos - pos[particleIndex]);
				neighborCounts[1]++;
			}

			// rule3: cohesion
			if (distance <= rule3Distance)
			{
				velocities[2] += vel[thisParticleIndex];
				neighborCounts[2]++;
			}
		}
	}
}

/*
 * helper function : coherent search within given grid
 * */
__device__ void computeVelocitiesWithinGivenGridCoherent(
	int gridIndex, int particleIndex,
	int* gridCellStartIndices, int* gridCellEndIndices,
	glm::vec3* pos, glm::vec3* vel,
	glm::vec3* velocities, int* neighborCounts)
{

	if (gridCellStartIndices[gridIndex] != -1 && gridCellEndIndices[gridIndex] != -1)
	{
		for (int i = gridCellStartIndices[gridIndex]; i <= gridCellEndIndices[gridIndex]; ++i)
		{
			if (i == particleIndex)
				continue;

			glm::vec3 thisParticlePos = pos[i];
			float distance = glm::length(thisParticlePos - pos[particleIndex]);

			// rule1: center of mass
			if (distance <= rule1Distance)
			{
				velocities[0] += thisParticlePos;
				neighborCounts[0]++;
			}

			// rule2: separation
			if (distance <= rule2Distance)
			{
				velocities[1] = velocities[1] - (thisParticlePos - pos[particleIndex]);
				neighborCounts[1]++;
			}

			// rule3: cohesion
			if (distance <= rule3Distance)
			{
				velocities[2] += vel[i];
				neighborCounts[2]++;
			}
		}
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

	int index = threadIdx.x + (blockDim.x * blockIdx.x);
	if (index >= N)
		return;

	int particleIndex = particleArrayIndices[index];
	glm::vec3 gridIndex3D = (pos[particleIndex] - gridMin) * inverseCellWidth;
	int x = int(gridIndex3D.x);
	int y = int(gridIndex3D.y);
	int z = int(gridIndex3D.z);

	int x_diff = (int(gridIndex3D.x + 0.5f) - x) == 0 ? -1 : 1;
	int y_diff = (int(gridIndex3D.y + 0.5f) - y) == 0 ? -1 : 1;
	int z_diff = (int(gridIndex3D.z + 0.5f) - z) == 0 ? -1 : 1;
	
	// new velocities by different rules
	glm::vec3 velocities[3];
	int neighborCounts[3];
	for (int i = 0; i < 3; ++i)
	{
		velocities[i] = glm::vec3(0, 0, 0);
		neighborCounts[i] = 0;
	}

	//int gridIndex1D;
	int interestedGridCellIndices[8];
	interestedGridCellIndices[0] = gridIndex3Dto1D(x,		   y,		   z		 , gridResolution);
	interestedGridCellIndices[1] = gridIndex3Dto1D(x + x_diff, y, 		   z		 , gridResolution);
	interestedGridCellIndices[2] = gridIndex3Dto1D(x, 		   y + y_diff, z		 , gridResolution);
	interestedGridCellIndices[3] = gridIndex3Dto1D(x, 		   y,		   z + z_diff, gridResolution);
	interestedGridCellIndices[4] = gridIndex3Dto1D(x + x_diff, y + y_diff, z		 , gridResolution);
	interestedGridCellIndices[5] = gridIndex3Dto1D(x, 		   y + y_diff, z + z_diff, gridResolution);
	interestedGridCellIndices[6] = gridIndex3Dto1D(x + x_diff, y, 		   z + z_diff, gridResolution);
	interestedGridCellIndices[7] = gridIndex3Dto1D(x + x_diff, y + y_diff, z + z_diff, gridResolution);

	int gridCellCount = gridResolution * gridResolution * gridResolution;
	int gridIndex;
	for(int i=0;i<8;++i)
	{
		gridIndex = interestedGridCellIndices[i];
		if(gridIndex >=0 && gridIndex < gridCellCount)
		{
			computeVelocitiesWithinGivenGrid(
				gridIndex, particleIndex,
				gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
				pos, vel1,
				velocities, neighborCounts);
		}
	}

	/* ugly codes
	// for (x,y,z) neighborhood
	gridIndex1D = gridIndex3Dto1D(x, y, z, gridResolution);
	if (gridIndex1D >= 0 && gridIndex1D < gridCellCount)
	{
		computeVelocitiesWithinGivenGrid(
			gridIndex1D, particleIndex,
			gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
			pos, vel1,
			velocities, neighborCounts);
	}

	// for (x+x_diff,y,z) neighborhood
	gridIndex1D = gridIndex3Dto1D(x + x_diff, y, z, gridResolution);
	if (gridIndex1D >= 0 && gridIndex1D < gridCellCount)
	{
		computeVelocitiesWithinGivenGrid(
			gridIndex1D, particleIndex,
			gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
			pos, vel1,
			velocities, neighborCounts);
	}


	// for (x,y+y_diff,z) neighborhood
	gridIndex1D = gridIndex3Dto1D(x, y + y_diff, z, gridResolution);
	if (gridIndex1D >= 0 && gridIndex1D < gridCellCount)
	{
		computeVelocitiesWithinGivenGrid(
			gridIndex1D, particleIndex,
			gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
			pos, vel1,
			velocities, neighborCounts);
	}

	// for (x,y,z+z_diff) neighborhood
	gridIndex1D = gridIndex3Dto1D(x, y, z + z_diff, gridResolution);
	if (gridIndex1D >= 0 && gridIndex1D < gridCellCount)
	{
		computeVelocitiesWithinGivenGrid(
			gridIndex1D, particleIndex,
			gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
			pos, vel1,
			velocities, neighborCounts);
	}

	// for (x+x_diff,y+y_diff,z) neighborhood
	gridIndex1D = gridIndex3Dto1D(x + x_diff, y + y_diff, z, gridResolution);
	if (gridIndex1D >= 0 && gridIndex1D < gridCellCount)
	{
		computeVelocitiesWithinGivenGrid(
			gridIndex1D, particleIndex,
			gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
			pos, vel1,
			velocities, neighborCounts);
	}

	// for (x+x_diff,y,z+z_diff) neighborhood
	gridIndex1D = gridIndex3Dto1D(x + x_diff, y, z + z_diff, gridResolution);
	if (gridIndex1D >= 0 && gridIndex1D < gridCellCount)
	{
		computeVelocitiesWithinGivenGrid(
			gridIndex1D, particleIndex,
			gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
			pos, vel1,
			velocities, neighborCounts);
	}


	// for (x,y+y_diff,z+z_diff) neighborhood
	gridIndex1D = gridIndex3Dto1D(x, y + y_diff, z + z_diff, gridResolution);
	if (gridIndex1D >= 0 && gridIndex1D < gridCellCount)
	{
		computeVelocitiesWithinGivenGrid(
			gridIndex1D, particleIndex,
			gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
			pos, vel1,
			velocities, neighborCounts);
	}

	// for (x+x_diff,y+y_diff,z+z_diff) neighborhood
	gridIndex1D = gridIndex3Dto1D(x + x_diff, y + y_diff, z + z_diff, gridResolution);
	if (gridIndex1D >= 0 && gridIndex1D < gridCellCount)
	{
		computeVelocitiesWithinGivenGrid(
			gridIndex1D, particleIndex,
			gridCellStartIndices, gridCellEndIndices, particleArrayIndices,
			pos, vel1,
			velocities, neighborCounts);
	}
	*/

	// compute new speed
	// rule1 center of mass
	if (neighborCounts[0] > 0)
	{
		velocities[0] = velocities[0] / float(neighborCounts[0]);
		velocities[0] = (velocities[0] - pos[particleIndex]) * rule1Scale;
	}

	// rule2 separation
	velocities[1] = velocities[1] * rule2Scale;

	// rule3 cohesion
	if (neighborCounts[2] > 0)
	{
		velocities[2] = velocities[2] / float(neighborCounts[2]);
		velocities[2] = (velocities[2] - vel1[particleIndex]) * rule3Scale;
	}

	// sum them up
	glm::vec3 newVel = vel1[particleIndex];
	for (int i = 0; i < 3; ++i)
	{
		newVel += velocities[i];
	}

	// clamp new volocity
	if (glm::length(newVel) >= maxSpeed)
	{
		newVel = glm::normalize(newVel) * maxSpeed;
	}

	// write back newVel to vel2
	vel2[particleIndex] = newVel;
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

	int index = threadIdx.x + (blockDim.x * blockIdx.x);
	if (index >= N)
		return;

	glm::vec3 gridIndex3D = (pos[index] - gridMin) * inverseCellWidth;
	int x = int(gridIndex3D.x);
	int y = int(gridIndex3D.y);
	int z = int(gridIndex3D.z);

	int x_diff = (int(gridIndex3D.x + 0.5f) - x) == 0 ? -1 : 1;
	int y_diff = (int(gridIndex3D.y + 0.5f) - y) == 0 ? -1 : 1;
	int z_diff = (int(gridIndex3D.z + 0.5f) - z) == 0 ? -1 : 1;

	// new velocities by different rules
	glm::vec3 velocities[3];
	int neighborCounts[3];
	for (int i = 0; i < 3; ++i)
	{
		velocities[i] = glm::vec3(0, 0, 0);
		neighborCounts[i] = 0;
	}

	//int gridIndex1D;
	int interestedGridCellIndices[8];
	interestedGridCellIndices[0] = gridIndex3Dto1D(x,		   y,		   z		 , gridResolution);
	interestedGridCellIndices[1] = gridIndex3Dto1D(x + x_diff, y, 		   z		 , gridResolution);
	interestedGridCellIndices[2] = gridIndex3Dto1D(x, 		   y + y_diff, z		 , gridResolution);
	interestedGridCellIndices[3] = gridIndex3Dto1D(x, 		   y,		   z + z_diff, gridResolution);
	interestedGridCellIndices[4] = gridIndex3Dto1D(x + x_diff, y + y_diff, z		 , gridResolution);
	interestedGridCellIndices[5] = gridIndex3Dto1D(x, 		   y + y_diff, z + z_diff, gridResolution);
	interestedGridCellIndices[6] = gridIndex3Dto1D(x + x_diff, y, 		   z + z_diff, gridResolution);
	interestedGridCellIndices[7] = gridIndex3Dto1D(x + x_diff, y + y_diff, z + z_diff, gridResolution);

	int gridCellCount = gridResolution * gridResolution * gridResolution;
	int gridIndex;
	for(int i=0;i<8;++i)
	{
		gridIndex = interestedGridCellIndices[i];
		if(gridIndex >=0 && gridIndex < gridCellCount)
		{
			computeVelocitiesWithinGivenGridCoherent(
				gridIndex, index,
				gridCellStartIndices, gridCellEndIndices,
				pos, vel1,
				velocities, neighborCounts);
		}
	}

	// compute new speed
	// rule1 center of mass
	if (neighborCounts[0] > 0)
	{
		velocities[0] = velocities[0] / float(neighborCounts[0]);
		velocities[0] = (velocities[0] - pos[index]) * rule1Scale;
	}

	// rule2 separation
	velocities[1] = velocities[1] * rule2Scale;

	// rule3 cohesion
	if (neighborCounts[2] > 0)
	{
		velocities[2] = velocities[2] / float(neighborCounts[2]);
		velocities[2] = (velocities[2] - vel1[index]) * rule3Scale;
	}

	// sum them up
	glm::vec3 newVel = vel1[index];
	for (int i = 0; i < 3; ++i)
	{
		newVel += velocities[i];
	}

	// clamp new volocity
	if (glm::length(newVel) >= maxSpeed)
	{
		newVel = glm::normalize(newVel) * maxSpeed;
	}

	// write back newVel to vel2
	vel2[index] = newVel;
}

/*
 * rearrange pos and vel data according to particleArrayIndices
 * */
__global__ void kernReshufflePosAndVelArray(
	int N,
	int* particleArrayIndices,
	glm::vec3* pos, glm::vec3* vel,
	glm::vec3* shuffledPos, glm::vec3* shuffledVel)
{

	int index = threadIdx.x + (blockDim.x * blockIdx.x);
	if(index >= N)
		return;

	shuffledPos[index] = pos[particleArrayIndices[index]];
	shuffledVel[index] = vel[particleArrayIndices[index]];
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {

  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernUpdateVelocityBruteForce << < fullBlocksPerGrid, blockSize >> >(
		numObjects, dev_pos, 
		bActiveDevice ? dev_vel2 : dev_vel1, 
		bActiveDevice ? dev_vel1 : dev_vel2);

	// update position 
	kernUpdatePos << < fullBlocksPerGrid, blockSize >> >(
		numObjects, dt, dev_pos,
		bActiveDevice ? dev_vel1 : dev_vel2);

  // TODO-1.2 ping-pong the velocity buffers
	bActiveDevice = !bActiveDevice;
		
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

	// label each boid, **gridResolution = gridSideCount**
	kernComputeIndices << <fullBlocksPerGrid, blockSize >> >(
		numObjects, gridSideCount, 
		gridMinimum, gridInverseCellWidth, 
		dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

	// sort according to gridIndex
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

	// find start and end indices
	// 1.reset CellStart and CellEnd to -1, which means no boid encloses with any boids
	kernResetIntBuffer << < (gridCellCount + blockSize - 1) / blockSize, blockSize >> >(gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << < (gridCellCount + blockSize - 1) / blockSize, blockSize >> >(gridCellCount, dev_gridCellEndIndices, -1);

	// 2.Identify start and end indices for each grid cell
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> >(
		numObjects,
		dev_particleGridIndices,
		dev_gridCellStartIndices, dev_gridCellEndIndices);

	// update velocity by neighbor search scattered
	kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> >(
		numObjects, gridSideCount, gridMinimum,
		gridInverseCellWidth, gridCellWidth,
		dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_particleArrayIndices,
		dev_pos, 
		bActiveDevice ? dev_vel2 : dev_vel1,
		bActiveDevice ? dev_vel1 : dev_vel2);

	// update position 
	kernUpdatePos << < fullBlocksPerGrid, blockSize >> >(
		numObjects, dt, dev_pos,
		bActiveDevice ? dev_vel1 : dev_vel2);

	// ping-pong the velocity buffers
	bActiveDevice = !bActiveDevice;

	////////////////////////////////////////////////////////////////////////
	//!!! FOR DEBUGGING !!!//
	#define __DEBUG__ 0
	#if __DEBUG__
	int* startIdx = new int[gridCellCount];
	int* endIdx = new int[gridCellCount];
	int* gridIdx = new int[numObjects];

	cudaMemcpy(gridIdx, dev_particleGridIndices, numObjects * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(startIdx, dev_gridCellStartIndices, gridCellCount * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(endIdx, dev_gridCellEndIndices, gridCellCount * sizeof(int), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < numObjects; ++i)
	//{
	//	std::cout << gridIdx[i] << std::endl;
	//}

	//int count = 0;
	//for (int i = 0; i < gridCellCount; ++i)
	//{
	//	if (startIdx[i] != -1 && endIdx[i] != -1)
	//	{
	//		//std::cout
	//		//	<< "i=" << i << " => "
	//		//	<< "startidx = " << startIdx[i] << ", endidx = " << endIdx[i] << std::endl;
	//		count += (endIdx[i] - startIdx[i] + 1);
	//		if (count == numObjects)
	//			std::cout << "total numObjects => i = " << i << std::endl;
	//	}
	//}
	//std::cout << count << std::endl;
	//std::cout << "final grid index = " <<gridIdx[numObjects - 1] << std::endl;

	delete[] startIdx;
	delete[] endIdx;
	delete[] gridIdx;
	#endif
	////////////////////////////////////////////////////////////////////////
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

	// label each boid, **gridResolution = gridSideCount**
	kernComputeIndices << <fullBlocksPerGrid, blockSize >> >(
		numObjects, gridSideCount,
		gridMinimum, gridInverseCellWidth,
		dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

	// sort according to gridIndex
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

	// find start and end indices
	// 1.reset CellStart and CellEnd to -1, which means no boid encloses with any boids
	kernResetIntBuffer << < (gridCellCount + blockSize - 1) / blockSize, blockSize >> >(gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << < (gridCellCount + blockSize - 1) / blockSize, blockSize >> >(gridCellCount, dev_gridCellEndIndices, -1);

	// 2.Identify start and end indices for each grid cell
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> >(
		numObjects,
		dev_particleGridIndices,
		dev_gridCellStartIndices, dev_gridCellEndIndices);

	// re-shuffle all particle data in simulation array
 	glm::vec3* vel = bActiveDevice ? dev_vel2 : dev_vel1; // get clean velocity data pointer

 	kernReshufflePosAndVelArray << <fullBlocksPerGrid, blockSize >> >(
 		numObjects,
 		dev_particleArrayIndices,
 		dev_pos, vel,
 		dev_shuffledPos, dev_shuffledVel);

 	glm::vec3 *tmp;
 	// swap pos array pointer
 	tmp = dev_pos;
 	dev_pos = dev_shuffledPos;
 	dev_shuffledPos = tmp;

 	// swap vel array pointer
 	if(bActiveDevice) // dev_vel2 is clean
 	{
 		tmp = dev_vel2;
 		dev_vel2 = dev_shuffledVel;
 		dev_shuffledVel = tmp;
 	}
 	else // dev_vel1 is clean
	{
		tmp = dev_vel1;
		dev_vel1 = dev_shuffledVel;
		dev_shuffledVel = tmp;
	}

 	// update velocity
 	kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> >(
 		numObjects, gridSideCount, gridMinimum,
 		gridInverseCellWidth, gridCellWidth,
 	    dev_gridCellStartIndices, dev_gridCellEndIndices,
 	    dev_pos,
 	    bActiveDevice ? dev_vel2 : dev_vel1,
 		bActiveDevice ? dev_vel1 : dev_vel2);


	// update position
	kernUpdatePos << < fullBlocksPerGrid, blockSize >> >(
		numObjects, dt, dev_pos,
		bActiveDevice ? dev_vel1 : dev_vel2);

	// ping-pong the velocity buffers
	bActiveDevice = !bActiveDevice;
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

  cudaFree(dev_shuffledPos);
  cudaFree(dev_shuffledVel);

  checkCUDAErrorWithLine("cuda free failed!");
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
  delete(intKeys);
  delete(intValues);
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
