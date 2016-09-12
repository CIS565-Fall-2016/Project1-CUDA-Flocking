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

#define DEBUG 0

#define PROFILE 1

#if PROFILE 
//Events for timing analysis
cudaEvent_t beginLoop;
cudaEvent_t endLoop;
cudaEvent_t beginEvent;
cudaEvent_t endEvent;

//event time records
float randomPosKernelTime;
float searchAlgoTime;
#endif

#if DEBUG
#define NUMBOIDS 10
int printcnt = 0;
int maxprints = 4;
#endif

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

#define maxVel 1.0f
#define minVel -1.0f

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
glm::vec3 *dev_orderedPos;
glm::vec3 *dev_orderedVel;

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
#if PROFILE
  cudaEventCreate(&beginEvent);
  cudaEventCreate(&endEvent);
#endif

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");
  
#if PROFILE
  cudaEventRecord(beginEvent, 0);
#endif

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

#if PROFILE
  cudaEventRecord(endEvent, 0);
  cudaEventSynchronize(endEvent);
  cudaEventElapsedTime(&randomPosKernelTime, beginEvent, endEvent);
  std::cout << "pos init Time: " << randomPosKernelTime << std::endl;
#endif

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

  cudaMalloc((void**)&dev_orderedPos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_orderedPos failed!");

  cudaMalloc((void**)&dev_orderedVel, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_orderedVel failed!");

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
	glm::vec3 centerOfMass = glm::vec3(0.0f, 0.0f, 0.0f); //rule 1
	glm::vec3 keepAway     = glm::vec3(0.0f, 0.0f, 0.0f); //rule 2
	glm::vec3 neighborVels = glm::vec3(0.0f, 0.0f, 0.0f); //rule 3

	int cnt1 = 0;
	int cnt3 = 0;

	for (int iBoid = 0; iBoid < N; ++iBoid)
	{
		if (iBoid == iSelf) continue;


		// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
		if (glm::length(pos[iBoid] - pos[iSelf]) < rule1Distance)
		{
			centerOfMass = centerOfMass + pos[iBoid];
			++cnt1;
		}
		// Rule 2: boids try to stay a distance d away from each other
		if (glm::length(pos[iBoid] - pos[iSelf]) < rule2Distance)
			keepAway = keepAway - (pos[iBoid] - pos[iSelf]);

		// Rule 3: boids try to match the speed of surrounding boids
		if (glm::length(pos[iBoid] - pos[iSelf]) < rule3Distance)
		{
			neighborVels = neighborVels + vel[iBoid];
			++cnt3;
		}
	}

	//calculate averaged parameters
	if (cnt1) centerOfMass = (centerOfMass / (float) cnt1 - pos[iSelf]) * rule1Scale;
	keepAway = keepAway * rule2Scale;
	if (cnt3) neighborVels = (neighborVels / (float) cnt3 - vel[iSelf]) * rule3Scale;
	
	return vel[iSelf] + centerOfMass + keepAway + neighborVels;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {

	//calculate index
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}

	// Compute a new velocity based on pos and vel1
	glm::vec3 newVel = computeVelocityChange(N, index, pos, vel1);
	
	// Clamp the speed
	newVel.x = glm::clamp(newVel.x, minVel, maxVel);
	newVel.y = glm::clamp(newVel.y, minVel, maxVel);
	newVel.z = glm::clamp(newVel.z, minVel, maxVel);

	// Record the new velocity into vel2. Question: why NOT vel1? Next result depends on prev vels 
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



__device__ glm::vec3 posToFloat3DIndex(glm::vec3 pos, glm::vec3 gridMin, float inverseCellWidth)
{
	//to zero-index everything, must subtract off minimum value
	//NOTE these are still floats!!
	return  glm::vec3(((pos.x - gridMin.x) * inverseCellWidth),
		((pos.y - gridMin.y) * inverseCellWidth),
		((pos.z - gridMin.z) * inverseCellWidth));
}


__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) 
	{
		return;
	}
    // Label each boid with the index of its grid cell.
	glm::vec3 grid3DIndex = posToFloat3DIndex(pos[index], gridMin, inverseCellWidth);
	int gridCell = gridIndex3Dto1D((int)grid3DIndex.x, (int)grid3DIndex.y, (int)grid3DIndex.z, gridResolution);

#if 0
	if (index == 0){
		printf("my index: %d\n my cell: %d\n", index, gridCell);
		printf("my pos: %f %f %f\n", pos[index].x, pos[index].y, pos[index].z);
		printf("my 3D grid: %f %f %f\n", grid3DIndex.x, grid3DIndex.y, grid3DIndex.z);
		printf("my gridcell: %d\n", gridCell);
	}
#endif

	gridIndices[index] = gridCell; //index is boid index, points to grid index


    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
	indices[index] = index; //index corresponds to gridIndices indices, points to boid index
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

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N)
	{
		return;
	}

	// Identify the start point of each cell in the gridIndices array.
	// This is basically a parallel unrolling of a loop that goes
	// "this index doesn't match the one before it, must be a new cell!"
	int myCell = particleGridIndices[index];

	if (index == 0 || particleGridIndices[index - 1] != myCell)
	{
		gridCellStartIndices[myCell] = index;
	}

	if (index == N-1 || particleGridIndices[index + 1] != myCell)
	{
		gridCellEndIndices[myCell] = index;
	}
}


__device__ int getNeighbors(glm::vec3 pos, float inverseCellWidth, 
	float cellWidth, int gridResolution, glm::vec3 gridMin, int * neighbors)
{

	float halfWidth = cellWidth * 0.5f;
	glm::vec3 myFloatGridPos =  posToFloat3DIndex (pos, gridMin, inverseCellWidth);

	glm::vec3 gridStart = glm::vec3( 0.0f, 0.0f, 0.0f );
	glm::vec3 gridEnd = glm::vec3( 0.0f, 0.0f, 0.0f );

	//if adding a halfwidth results in the same tile, then they are in 
	if ((int)((pos.x - gridMin.x + halfWidth) * inverseCellWidth) == (int)myFloatGridPos.x)
		gridStart.x = -1.0f ;
	else 
		gridEnd.x = 1.0f ;
	
	if ((int)((pos.y - gridMin.y + halfWidth) * inverseCellWidth) == (int)myFloatGridPos.y)
		gridStart.y = -1.0f ;
	else 
		gridEnd.y = 1.0f ;

	if ((int)((pos.z - gridMin.z + halfWidth) * inverseCellWidth) == (int)myFloatGridPos.z)
		gridStart.z = -1.0f ;
	else 
		gridEnd.z = 1.0f ;

	//calculate which cells are adjacent to me and put them in the buffer
	int neighborCnt = 0; 

	for (int i = (int)myFloatGridPos.x + (int)gridStart.x; i <= (int)myFloatGridPos.x + (int)gridEnd.x; ++i)
	{

		if (i < 0 || i >= gridResolution)
			continue;

		for (int j = (int)myFloatGridPos.y + (int)gridStart.y; j <= (int)myFloatGridPos.y + (int)gridEnd.y; ++j)
		{
			if (j < 0 || j >= gridResolution)
			continue;

			for (int k = (int)myFloatGridPos.z + (int)gridStart.z; k <= (int)myFloatGridPos.z + (int)gridEnd.z; ++k)
			{

				if (k < 0 || k >= gridResolution)
					continue;

				int neighborCell = gridIndex3Dto1D(i, j, k, gridResolution); 

				neighbors[neighborCnt] = neighborCell;

				++ neighborCnt;
			}
		}
	}

	return neighborCnt ;
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {

	// Update a boid's velocity using the uniform grid to reduce
	// the number of boids that need to be checked.
	
	int particleNum = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (particleNum >= N)
	{
		return;
	}

	int myBoidIndex = particleArrayIndices[particleNum];

	//Get a list of the grid cells that this particle is in
	//and its closest relevant neighbors
	int neighbors[8];
	int neighborCnt = getNeighbors(pos[myBoidIndex],
						inverseCellWidth, cellWidth, gridResolution, gridMin, neighbors);

#if DEBUG
	if (myBoidIndex == 10) { for (int d = 0; d < neighborCnt; ++d) printf("neighbor %d = %d\n", d, neighbors[d]); }
#endif

	glm::vec3 centerOfMass = glm::vec3(0.0f, 0.0f, 0.0f); //rule 1
	glm::vec3 keepAway = glm::vec3(0.0f, 0.0f, 0.0f); //rule 2
	glm::vec3 neighborVels = glm::vec3(0.0f, 0.0f, 0.0f); //rule 3

	int cnt1 = 0;
	int cnt3 = 0;
	
	for (int i = 0; i < neighborCnt; ++i)
	{
		// For each cell, read the start/end indices in the boid pointer array.
		int currentCellIndex = neighbors[i];
		int startIndex = gridCellStartIndices[currentCellIndex];
		int endIndex = gridCellEndIndices[currentCellIndex];

#if DEBUG
		if (myBoidIndex == 10) { printf("start %d end %d\n", startIndex, endIndex); }
#endif

		// Access each boid in the cell and compute velocity change from
		// the boids rules, if this boid is within the neighborhood distance.
		for (int iterIndex = startIndex; iterIndex <= endIndex; ++iterIndex)
		{
			int neighborBoidIndex = particleArrayIndices[iterIndex];

			if (myBoidIndex == neighborBoidIndex) continue;

			// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
			if (glm::length(pos[neighborBoidIndex] - pos[myBoidIndex]) < rule1Distance)
			{
				centerOfMass = centerOfMass + pos[neighborBoidIndex];
				++cnt1;
			}

			// Rule 2: boids try to stay a distance d away from each other
			if (glm::length(pos[neighborBoidIndex] - pos[myBoidIndex]) < rule2Distance)
				keepAway = keepAway - (pos[neighborBoidIndex] - pos[myBoidIndex]);

			// Rule 3: boids try to match the speed of surrounding boids
			if (glm::length(pos[neighborBoidIndex] - pos[myBoidIndex]) < rule3Distance)
			{
				neighborVels = neighborVels + vel1[neighborBoidIndex];
				++cnt3;
			}
		}
	}

	//calculate averaged parameters
	if (cnt1) centerOfMass = (centerOfMass / (float)cnt1 - pos[myBoidIndex]) * rule1Scale;
	keepAway = keepAway * rule2Scale;
	if (cnt3) neighborVels = (neighborVels / (float)cnt3 - vel1[myBoidIndex]) * rule3Scale;

	glm::vec3 newVel = vel1[myBoidIndex] + centerOfMass + keepAway + neighborVels;

#if DEBUG
	if (myBoidIndex == 10){
		printf("my pos is %f %f %f\n", pos[10].x, pos[10].y, pos[10].z);
		printf("cnt1= %d, cnt3=%d\n", cnt1, cnt3);
		printf("newvel is %f %f %f\n", newVel.x, newVel.y, newVel.z);
	}
#endif

	// Clamp the speed change before putting the new speed in vel2
	newVel.x = glm::clamp(newVel.x, minVel, maxVel);
	newVel.y = glm::clamp(newVel.y, minVel, maxVel);
	newVel.z = glm::clamp(newVel.z, minVel, maxVel);

	vel2[myBoidIndex] = newVel; 
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
	int particleIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (particleIndex >= N)
	{
		return;
	}

  // - Identify which cells may contain neighbors. This isn't always 8.
	//Get a list of the grid cells that this particle is in
	//and its closest relevant neighbors
	int neighbors[8];
	int neighborCnt = getNeighbors(pos[particleIndex],
		inverseCellWidth, cellWidth, gridResolution, gridMin, neighbors);

#if DEBUG
	if (particleIndex == 10) { for (int d = 0; d < neighborCnt; ++d) printf("neighbor %d = %d\n", d, neighbors[d]); }
#endif

	glm::vec3 centerOfMass = glm::vec3(0.0f, 0.0f, 0.0f); //rule 1
	glm::vec3 keepAway = glm::vec3(0.0f, 0.0f, 0.0f); //rule 2
	glm::vec3 neighborVels = glm::vec3(0.0f, 0.0f, 0.0f); //rule 3

	int cnt1 = 0;
	int cnt3 = 0;

	for (int i = 0; i < neighborCnt; ++i)
	{
		// - For each cell, read the start/end indices in the boid pointer array.
		//   DIFFERENCE: For best results, consider what order the cells should be
		//   checked in to maximize the memory benefits of reordering the boids data.
		int currentCellIndex = neighbors[i];
		int startIndex = gridCellStartIndices[currentCellIndex];
		int endIndex = gridCellEndIndices[currentCellIndex];
#if DEBUG
		if (particleIndex == 10) { printf("start %d end %d\n", startIndex, endIndex); }
#endif

		// - Access each boid in the cell and compute velocity change from
		//   the boids rules, if this boid is within the neighborhood distance.
		for (int neighborIndex = startIndex; neighborIndex <= endIndex; ++neighborIndex)
		{

			if (neighborIndex == particleIndex) continue;

			// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
			if (glm::length(pos[neighborIndex] - pos[particleIndex]) < rule1Distance)
			{
				centerOfMass = centerOfMass + pos[neighborIndex];
				++cnt1;
			}

			// Rule 2: boids try to stay a distance d away from each other
			if (glm::length(pos[neighborIndex] - pos[particleIndex]) < rule2Distance)
				keepAway = keepAway - (pos[neighborIndex] - pos[particleIndex]);

			// Rule 3: boids try to match the speed of surrounding boids
			if (glm::length(pos[neighborIndex] - pos[particleIndex]) < rule3Distance)
			{
				neighborVels = neighborVels + vel1[neighborIndex];
				++cnt3;
			}
		}

	}

	//calculate averaged parameters
	if (cnt1) centerOfMass = (centerOfMass / (float)cnt1 - pos[particleIndex]) * rule1Scale;
	keepAway = keepAway * rule2Scale;
	if (cnt3) neighborVels = (neighborVels / (float)cnt3 - vel1[particleIndex]) * rule3Scale;

	glm::vec3 newVel = vel1[particleIndex] + centerOfMass + keepAway + neighborVels;

#if DEBUG
	if (particleIndex == 10){
		printf("my pos is %f %f %f\n", pos[10].x, pos[10].y, pos[10].z);
		printf("cnt1= %d, cnt3=%d\n", cnt1, cnt3);
		printf("newvel is %f %f %f\n", newVel.x, newVel.y, newVel.z);
	}
#endif

	// - Clamp the speed change before putting the new speed in vel2
	newVel.x = glm::clamp(newVel.x, minVel, maxVel);
	newVel.y = glm::clamp(newVel.y, minVel, maxVel);
	newVel.z = glm::clamp(newVel.z, minVel, maxVel);

	vel2[particleIndex] = newVel;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
	dim3 fullBlocksPerGrid = (numObjects + blockSize - 1) / blockSize ;

#if PROFILE
	cudaEventRecord(beginEvent, 0);
#endif
	// TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
	kernUpdateVelocityBruteForce <<<fullBlocksPerGrid, blockSize >>>(numObjects, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");
#if PROFILE
	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&searchAlgoTime, beginEvent, endEvent);
	std::cout << "search Time: " << searchAlgoTime << std::endl;
#endif
	kernUpdatePos <<<fullBlocksPerGrid, blockSize >>>(numObjects, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("kernUpdatePos failed!");
	
	// TODO-1.2 ping-pong the velocity buffers
	
	glm::vec3 *tmp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = tmp;
}

void Boids::stepSimulationScatteredGrid(float dt) {

  // Uniform Grid Neighbor search using Thrust sort.

	dim3 fullBlocksPerGrid = (numObjects + blockSize - 1) / blockSize;
	dim3 fullBlocksPerGridForCells = (gridCellCount + blockSize - 1) / blockSize;

#if DEBUG
	glm::vec3 pos[NUMBOIDS];

	if (printcnt < maxprints){

		cudaMemcpy(pos, dev_pos, sizeof(glm::vec3) * NUMBOIDS, cudaMemcpyDeviceToHost);

		std::cout << "positions: " << std::endl;
		for (int i = 0; i < NUMBOIDS; i++) {
			std::cout << " boid#: " << i;
			std::cout << " pos : " << pos[i].x << " " << pos[i].y << " " << pos[i].z << std::endl;
		}
	}
#endif

  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.

	kernComputeIndices <<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount,
		gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices); 
	checkCUDAErrorWithLine("kernComputeIndices failed!");

  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
	dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
	dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);

#if DEBUG
	int particleGridIndices[NUMBOIDS];
	int particleArrayIndices[NUMBOIDS];

	if (printcnt < maxprints){

		cudaMemcpy(particleGridIndices, dev_particleGridIndices, sizeof(int) * NUMBOIDS, cudaMemcpyDeviceToHost);
		cudaMemcpy(particleArrayIndices, dev_particleArrayIndices, sizeof(int) * NUMBOIDS, cudaMemcpyDeviceToHost);

		std::cout << "thrust: before unstable sort: " << std::endl;
		for (int i = 0; i < NUMBOIDS; i++) {
			std::cout << "  key: " << particleGridIndices[i];
			std::cout << " value: " << particleArrayIndices[i] << std::endl;
		}
	}
#endif

	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, 
						dev_thrust_particleArrayIndices);

#if DEBUG
	if (printcnt < maxprints){
		cudaMemcpy(particleGridIndices, dev_particleGridIndices, sizeof(int) * NUMBOIDS, cudaMemcpyDeviceToHost);
		cudaMemcpy(particleArrayIndices, dev_particleArrayIndices, sizeof(int) * NUMBOIDS, cudaMemcpyDeviceToHost);

		std::cout << "thrust: after unstable sort: " << std::endl;
		for (int i = 0; i < NUMBOIDS; i++) {
			std::cout << "  key: " << particleGridIndices[i];
			std::cout << " value: " << particleArrayIndices[i] << std::endl;
		}
	}

#endif

	kernResetIntBuffer << <fullBlocksPerGridForCells, blockSize >> >(gridCellCount, dev_gridCellStartIndices, -1);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd1 failed!");
	kernResetIntBuffer << <fullBlocksPerGridForCells, blockSize >> >(gridCellCount, dev_gridCellEndIndices, -1);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd2 failed!");

  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
	kernIdentifyCellStartEnd << <fullBlocksPerGridForCells, blockSize >> > (numObjects, dev_particleGridIndices,
														dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

#if DEBUG
	const int cells = 22 * 22 * 22;
	int gridCellStartIndices[cells];
	int gridCellEndIndices[cells];

	if (printcnt < maxprints){
		cudaMemcpy(gridCellStartIndices, dev_gridCellStartIndices, sizeof(int) * cells, cudaMemcpyDeviceToHost);
		cudaMemcpy(gridCellEndIndices, dev_gridCellEndIndices, sizeof(int) * cells, cudaMemcpyDeviceToHost);

		std::cout << "start/end results: " << std::endl;
		for (int i = 0; i < cells; i++) {
			if (gridCellStartIndices[i] == -1 && gridCellEndIndices[i] == -1) continue;
			if (gridCellStartIndices[i] != -1 && gridCellEndIndices[i] != -1){

				std::cout << " cell index: " << i;
				std::cout << " start: " << gridCellStartIndices[i];
				std::cout << " end: " << gridCellEndIndices[i] << std::endl;
			}
			else
			{
				std::cout << "PROBLEM cell index: " << i;
				std::cout << " start: " << gridCellStartIndices[i];
				std::cout << " end: " << gridCellEndIndices[i] << std::endl;
			}
		}
	}

#endif

#if PROFILE
	cudaEventRecord(beginEvent, 0);
#endif

  // - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchScattered <<<fullBlocksPerGrid, blockSize >>> (
		numObjects, gridSideCount, gridMinimum,
		gridInverseCellWidth, gridCellWidth,
		dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_particleArrayIndices,
		dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");
#if PROFILE
	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&searchAlgoTime, beginEvent, endEvent);
	std::cout << "search Time: " << searchAlgoTime << std::endl;
#endif
  // - Update positions
	kernUpdatePos <<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("kernUpdatePos failed!");

  // - Ping-pong buffers as needed
	glm::vec3 *tmp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = tmp;

#if DEBUG
	printcnt++;
#endif
}

__global__ void kernRearrangeBoidData(
	int N, int *ordering, 
	glm::vec3 *originalPos, glm::vec3 *orderedPos, 
	glm::vec3 *originalVel, glm::vec3 *orderedVel) {

	int newIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (newIndex >= N)
	{
		return;
	}
	// boid at newIndex corresponds to pos and val at boidIndex
	int boidIndex = ordering[newIndex];

	// reorder data in new buffer to reflect newIndex
	orderedPos[newIndex] = originalPos[boidIndex];
	orderedVel[newIndex] = originalVel[boidIndex];
}

__global__ void kernReplaceBoidVelData(
	int N, int *ordering,
	glm::vec3 *originalVel, glm::vec3 *orderedVel) {

	int newIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (newIndex >= N)
	{
		return;
	}
	// boid at newIndex corresponds to pos and val at boidIndex
	int boidIndex = ordering[newIndex];

	// reorder data in new buffer to reflect newIndex
	originalVel[boidIndex] = orderedVel[newIndex];
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
	dim3 fullBlocksPerGrid = (numObjects + blockSize - 1) / blockSize;
	dim3 fullBlocksPerGridForCells = (gridCellCount + blockSize - 1) / blockSize;

  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
	kernComputeIndices <<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount,
		gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices); 
	checkCUDAErrorWithLine("kernComputeIndices failed!");

  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
	dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
	dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, 
		dev_thrust_particleArrayIndices);

	kernResetIntBuffer << <fullBlocksPerGridForCells, blockSize >> >(gridCellCount, dev_gridCellStartIndices, -1);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd1 failed!");
	kernResetIntBuffer << <fullBlocksPerGridForCells, blockSize >> >(gridCellCount, dev_gridCellEndIndices, -1);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd2 failed!");
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
	kernIdentifyCellStartEnd << <fullBlocksPerGridForCells, blockSize >> > (numObjects, dev_particleGridIndices,
		dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");


  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
	kernRearrangeBoidData << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_particleArrayIndices,
		dev_pos, dev_orderedPos, dev_vel1, dev_orderedVel);
	checkCUDAErrorWithLine("kernRearrangeBoidData failed!");

#if DEBUG
	int particleGridIndices[NUMBOIDS];
	int particleArrayIndices[NUMBOIDS];
	glm::vec3 originalpos[NUMBOIDS];
	glm::vec3 orderedpos[NUMBOIDS];
	glm::vec3 originalvel[NUMBOIDS];
	glm::vec3 orderedvel[NUMBOIDS];


	if (printcnt < maxprints){

		cudaMemcpy(particleGridIndices, dev_particleGridIndices, sizeof(int) * NUMBOIDS, cudaMemcpyDeviceToHost);
		cudaMemcpy(particleArrayIndices, dev_particleArrayIndices, sizeof(int) * NUMBOIDS, cudaMemcpyDeviceToHost);
		cudaMemcpy(originalpos, dev_pos, sizeof(glm::vec3) * NUMBOIDS, cudaMemcpyDeviceToHost);
		cudaMemcpy(orderedpos, dev_orderedPos, sizeof(glm::vec3) * NUMBOIDS, cudaMemcpyDeviceToHost);
		cudaMemcpy(originalvel, dev_vel1, sizeof(glm::vec3) * NUMBOIDS, cudaMemcpyDeviceToHost);
		cudaMemcpy(orderedvel, dev_orderedVel, sizeof(glm::vec3) * NUMBOIDS, cudaMemcpyDeviceToHost);

		std::cout << "PARTICLES: " << std::endl;
		for (int i = 0; i < NUMBOIDS; i++) {
			std::cout << "  particle index: " << i;
			std::cout << "  original boid index: " << particleArrayIndices[i];
			std::cout << "  grid index: " << particleGridIndices[i];
			std::cout << "  pos in original: " << originalpos[particleArrayIndices[i]].x << originalpos[particleArrayIndices[i]].y << originalpos[particleArrayIndices[i]].z;
			std::cout << "  pos in reordered: " << orderedpos[i].x << orderedpos[i].y << orderedpos[i].z;
			std::cout << "  vel in original: " << originalvel[particleArrayIndices[i]].x << originalvel[particleArrayIndices[i]].y << originalvel[particleArrayIndices[i]].z;
			std::cout << "  vel in reordered: " << orderedvel[i].x << orderedvel[i].y << orderedvel[i].z << std::endl;
		}
	}
#endif

#if PROFILE
	cudaEventRecord(beginEvent, 0);
#endif
  // - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> >(numObjects, gridSideCount, gridMinimum,
		gridInverseCellWidth, gridCellWidth,
		dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_orderedPos, dev_orderedVel, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");
#if PROFILE
	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&randomPosKernelTime, beginEvent, endEvent);
	std::cout << "search Time: " << searchAlgoTime << std::endl;
#endif
	//Replace the updated velocities in their original indices 
	kernReplaceBoidVelData << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_particleArrayIndices,
		dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernReplaceBoidVelData failed!");

  // - Update positions
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, dev_pos, dev_vel1);
	checkCUDAErrorWithLine("kernUpdatePos failed!");


  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
  // since we're using vel1 to hold the original ordering of the updated vel,
  // no need to ping-pong

#if DEBUG
	printcnt++;
#endif
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

  cudaFree(dev_orderedPos);
  cudaFree(dev_orderedVel);

#if PROFILE
  cudaEventDestroy(beginEvent);
  cudaEventDestroy(endEvent);
#endif
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
