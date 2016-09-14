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

// DONE-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3* dev_pos_shuffler;
glm::vec3* dev_vel_shuffler;

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
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
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

  // DONE-2.1 DONE-2.3 - Allocate additional buffers here.
  cudaMalloc((void **)&dev_particleArrayIndices, numObjects * sizeof(*dev_particleArrayIndices));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");
  cudaMalloc((void **)&dev_particleGridIndices, numObjects * sizeof(*dev_particleGridIndices));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
  
  cudaMalloc((void **)&dev_gridCellStartIndices, gridCellCount * sizeof(*dev_gridCellStartIndices));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");
  cudaMalloc((void **)&dev_gridCellEndIndices, gridCellCount * sizeof(*dev_gridCellEndIndices));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_pos_shuffler, numObjects * sizeof(*dev_pos_shuffler));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_shuffler failed!");
  cudaMalloc((void**)&dev_vel_shuffler, numObjects * sizeof(*dev_vel_shuffler));
  checkCUDAErrorWithLine("cudaMalloc dev_vel_shuffler failed!");

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
    glm::vec3 delta_vel(0.0f);

    glm::vec3 rule1_neighbor_pos_sum(0.0f);
    float rule1_neighbor_count = 0.0f;
    glm::vec3 rule2_total_offset(0.0f);
    glm::vec3 rule3_neighbor_vel_sum(0.0f);
    float rule3_neighbor_count = 0.0f;

    float current_distance;
    glm::vec3 current_pos_offset;
    for (int i = 0; i < N; i++)
    {
        if (i == iSelf){ continue; }

        current_pos_offset = pos[i] - pos[iSelf];
        current_distance = glm::length(current_pos_offset);

        // Rule 1: Get neighbor position sum and neighbor count for rule1
        if (current_distance < rule1Distance)
        {
            rule1_neighbor_pos_sum += pos[i];
            rule1_neighbor_count += 1.0f;
        }
        // Rule 2: Calculate offset for rule 2
        if (current_distance < rule2Distance)
        {
            rule2_total_offset -= current_pos_offset;
        }
        // Rule 3: Get velocity sum and neighbor count for rule 3
        if (current_distance < rule3Distance)
        {
            rule3_neighbor_vel_sum += vel[i];
            rule3_neighbor_count += 1.0f;
        }
    }

    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
    if (rule1_neighbor_count > 0.0f)
    {
        delta_vel += rule1Scale * ((rule1_neighbor_pos_sum / rule1_neighbor_count) - pos[iSelf]);
    }

    // Rule 2: boids try to stay a distance d away from each other
    delta_vel += rule2Scale *  rule2_total_offset;

    // Rule 3: boids try to match the speed of surrounding boids
    if (rule3_neighbor_count > 0.0f)
    {
        delta_vel += rule3Scale * ((rule3_neighbor_vel_sum / rule3_neighbor_count) - vel[iSelf]);
    }

    return delta_vel;
}

/**
* DONE-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2)
{
    // static const float max_sqr_speed = maxSpeed * maxSpeed;

    auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N)
    {
        return;
    }

    // Compute a new velocity based on pos and vel1
    glm::vec3 new_vel = vel1[index] + computeVelocityChange(N, index, pos, vel1);

    // Clamp the speed
    auto speed = glm::length(new_vel);
    if (speed > maxSpeed)
    {
        new_vel = new_vel * (maxSpeed / speed);
    }

    // Record the new velocity into vel2. Question: why NOT vel1?
    //   Ans: neighbors need vel from last frame to update their velocity
    vel2[index] = new_vel;
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
//   Ans: I think it is
//          for(z)
//            for(y)
//              for(x)
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
    return x + y * gridResolution + z * gridResolution * gridResolution;
}

// This function returns -1 if any dimension falls out of boundry
__device__ int gridIndex3Dto1DWithCheck(int x, int y, int z, int gridResolution)
{
    if (x < 0 || x >= gridResolution
        || y < 0 || y >= gridResolution
        || z < 0 || z >= gridResolution)
    {
        return -1;
    }
    else
    {
        return gridIndex3Dto1D(x, y, z, gridResolution);
    }
}

__device__ glm::vec3 gridIndex3DFromPosition(const glm::vec3& pos
    , const glm::vec3& grid_min, float inverse_cell_width)
{
    return (pos - grid_min) * inverse_cell_width;
}

__device__ int gridIndexFromPosition(const glm::vec3& pos, const glm::vec3& grid_min
    , float inverse_cell_width, int grid_resolution)
{
    auto gridindex_3d = gridIndex3DFromPosition(pos, grid_min, inverse_cell_width);
    auto gridindex = gridIndex3Dto1D(static_cast<int>(gridindex_3d.x)
        , static_cast<int>(gridindex_3d.y)
        , static_cast<int>(gridindex_3d.z)
        , grid_resolution);
	return gridindex;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices)
{
    // DONE-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N)
    {
        return;
    }

    indices[index] = index;
    gridIndices[index] = gridIndexFromPosition(pos[index], gridMin
        , inverseCellWidth, gridResolution);

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
    // DONE-2.1
    // Identify the start point of each cell in the gridIndices array.
    // This is basically a parallel unrolling of a loop that goes
    // "this index doesn't match the one before it, must be a new cell!"

    // index is sorted index, not boid index!
    auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N)
    {
        return;
    }

    auto prev_index = (index - 1 + N) % N;  // To minimize branches
    if (particleGridIndices[index] != particleGridIndices[prev_index])
    {
        gridCellStartIndices[particleGridIndices[index]] = index;
        gridCellEndIndices[particleGridIndices[prev_index]] = prev_index + 1;
          // intended to make the end side open like most C++ ranges
    }
}

// Helper function to determine direction (1 or -1) of the neighbor
// grid cell in one dimension
__device__ int directionOfNeighborSingleDimension(float axis_index)
{
    if (static_cast<int>(axis_index + 0.5f) > static_cast<int>(axis_index))
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

__global__ void kernUpdateVelNeighborSearchScattered(
    int N, int gridResolution, glm::vec3 gridMin
    , float inverseCellWidth, float cellWidth
    , int *gridCellStartIndices, int *gridCellEndIndices
    , int *particleArrayIndices, glm::vec3 *pos
	, glm::vec3 *vel1, glm::vec3 *vel2)
{
    // DONE-2.1 - Update a boid's velocity using the uniform grid to reduce
    // the number of boids that need to be checked.
    auto index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N)
    {
        return;
    }

    // - Identify the grid cell that this particle is in
    auto boid_index = particleArrayIndices[index];

	auto gridcell_index3d = gridIndex3DFromPosition(pos[boid_index], gridMin
		 , inverseCellWidth);

    // - Identify which cells may contain neighbors. This isn't always 8.
    auto sign_x = directionOfNeighborSingleDimension(gridcell_index3d.x);
    auto sign_y = directionOfNeighborSingleDimension(gridcell_index3d.y);
    auto sign_z = directionOfNeighborSingleDimension(gridcell_index3d.z);
    auto index_x = static_cast<int>(gridcell_index3d.x);
    auto index_y = static_cast<int>(gridcell_index3d.y);
    auto index_z = static_cast<int>(gridcell_index3d.z);

    int possible_neighbor_indices[8];
    possible_neighbor_indices[0] = gridIndex3Dto1DWithCheck(index_x, index_y, index_z
        , gridResolution);
    possible_neighbor_indices[1] = gridIndex3Dto1DWithCheck(index_x + sign_x, index_y, index_z
        , gridResolution);
    possible_neighbor_indices[2] = gridIndex3Dto1DWithCheck(index_x, index_y + sign_y, index_z
        , gridResolution);
    possible_neighbor_indices[3] = gridIndex3Dto1DWithCheck(index_x, index_y, index_z + sign_z
        , gridResolution);
    possible_neighbor_indices[4] = gridIndex3Dto1DWithCheck(index_x + sign_x, index_y + sign_y, index_z
        , gridResolution);
    possible_neighbor_indices[5] = gridIndex3Dto1DWithCheck(index_x + sign_x, index_y, index_z + sign_z
        , gridResolution);
    possible_neighbor_indices[6] = gridIndex3Dto1DWithCheck(index_x, index_y + sign_y, index_z + sign_z
        , gridResolution);
    possible_neighbor_indices[7] = gridIndex3Dto1DWithCheck(index_x + sign_x, index_y + sign_y, index_z + sign_z
        , gridResolution);


    // - For each cell, read the start/end indices in the boid pointer array.
    // - Access each boid in the cell and compute velocity change from
    //   the boids rules, if this boid is within the neighborhood distance.

    glm::vec3 delta_vel(0.0f);

    glm::vec3 rule1_neighbor_pos_sum(0.0f);
    float rule1_neighbor_count = 0.0f;
    glm::vec3 rule2_total_offset(0.0f);
    glm::vec3 rule3_neighbor_vel_sum(0.0f);
    float rule3_neighbor_count = 0.0f;

    float current_distance;
    glm::vec3 current_pos_offset;
    int iother;
    for (const auto neighbor_cell_index : possible_neighbor_indices)
    {
        if (neighbor_cell_index == -1)
        {
            continue;
        }

        for (int i = gridCellStartIndices[neighbor_cell_index]
            ; i < gridCellEndIndices[neighbor_cell_index]
            ; i++)
        {
            iother = particleArrayIndices[i];
            if (iother == boid_index){ continue; }

            current_pos_offset = pos[iother] - pos[boid_index];
            current_distance = glm::length(current_pos_offset);

            // Rule 1: Get neighbor position sum and neighbor count for rule1
            if (current_distance < rule1Distance)
            {
                rule1_neighbor_pos_sum += pos[iother];
                rule1_neighbor_count += 1.0f;
            }
            // Rule 2: Calculate offset for rule 2
            if (current_distance < rule2Distance)
            {
                rule2_total_offset -= current_pos_offset;
            }
            // Rule 3: Get velocity sum and neighbor count for rule 3
            if (current_distance < rule3Distance)
            {
                rule3_neighbor_vel_sum += vel1[iother];
                rule3_neighbor_count += 1.0f;
            }
        }
    }

    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
    if (rule1_neighbor_count > 0.0f)
    {
        delta_vel += rule1Scale * ((rule1_neighbor_pos_sum / rule1_neighbor_count) - pos[boid_index]);
    }

    // Rule 2: boids try to stay a distance d away from each other
    delta_vel += rule2Scale *  rule2_total_offset;

    // Rule 3: boids try to match the speed of surrounding boids
    if (rule3_neighbor_count > 0.0f)
    {
        delta_vel += rule3Scale * ((rule3_neighbor_vel_sum / rule3_neighbor_count) - vel1[boid_index]);
    }

    auto new_vel = vel1[boid_index] + delta_vel;

    // - Clamp the speed change before putting the new speed in vel2
    auto speed = glm::length(new_vel);
    if (speed > maxSpeed)
    {
        new_vel = new_vel * (maxSpeed / speed);
    }
    vel2[boid_index] = new_vel;

}

__global__ void kernReshuffle(int N, glm::vec3* pos
	, glm::vec3* vel, glm::vec3* pos_shuffler, glm::vec3* vel_shuffler
	, int* particleArrayIndices)
{
	// Reshuffle pos and vel using particleArrayIndices 
	// , store in pos_shuffler and vel_shuffler
	auto index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}

	pos_shuffler[index] = pos[particleArrayIndices[index]];
	vel_shuffler[index] = vel[particleArrayIndices[index]];
}

__global__ void kernUpdateVelNeighborSearchCoherent(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int *gridCellStartIndices, int *gridCellEndIndices,
	glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
	// DONE-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
	// except with one less level of indirection.
	// This should expect gridCellStartIndices and gridCellEndIndices to refer
	// directly to pos and vel1.

	auto index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}

	// - Identify the grid cell that this particle is in
	auto gridcell_index3d = gridIndex3DFromPosition(pos[index], gridMin
		, inverseCellWidth);

	// - Identify which cells may contain neighbors. This isn't always 8.
	auto sign_x = directionOfNeighborSingleDimension(gridcell_index3d.x);
	auto sign_y = directionOfNeighborSingleDimension(gridcell_index3d.y);
	auto sign_z = directionOfNeighborSingleDimension(gridcell_index3d.z);
	auto index_x = static_cast<int>(gridcell_index3d.x);
	auto index_y = static_cast<int>(gridcell_index3d.y);
	auto index_z = static_cast<int>(gridcell_index3d.z);

	int possible_neighbor_indices[8]; 
	// CHANGE: order of this array changed from 2.1 to match the best for(z): for(y): for(x) order
	possible_neighbor_indices[0] = gridIndex3Dto1DWithCheck(index_x, index_y, index_z
		, gridResolution);
	possible_neighbor_indices[1] = gridIndex3Dto1DWithCheck(index_x + sign_x, index_y, index_z
		, gridResolution);
	possible_neighbor_indices[2] = gridIndex3Dto1DWithCheck(index_x, index_y + sign_y, index_z
		, gridResolution);
	possible_neighbor_indices[3] = gridIndex3Dto1DWithCheck(index_x + sign_x, index_y + sign_y, index_z
		, gridResolution);
	possible_neighbor_indices[4] = gridIndex3Dto1DWithCheck(index_x, index_y, index_z + sign_z
		, gridResolution);
	possible_neighbor_indices[5] = gridIndex3Dto1DWithCheck(index_x + sign_x, index_y, index_z + sign_z
		, gridResolution);
	possible_neighbor_indices[6] = gridIndex3Dto1DWithCheck(index_x, index_y + sign_y, index_z + sign_z
		, gridResolution);
	possible_neighbor_indices[7] = gridIndex3Dto1DWithCheck(index_x + sign_x, index_y + sign_y, index_z + sign_z
		, gridResolution);

	// - For each cell, read the start/end indices in the boid pointer array.
	//   DIFFERENCE: For best results, consider what order the cells should be
	//   checked in to maximize the memory benefits of reordering the boids data.
	//   
	//   CHANGE: order of the array (possible_neighbor_indices) above already changed 
	//     so the order of the cells checked become
	//     for(z): for(y): for(x)
	glm::vec3 delta_vel(0.0f);

	glm::vec3 rule1_neighbor_pos_sum(0.0f);
	float rule1_neighbor_count = 0.0f;
	glm::vec3 rule2_total_offset(0.0f);
	glm::vec3 rule3_neighbor_vel_sum(0.0f);
	float rule3_neighbor_count = 0.0f;

	float current_distance;
	glm::vec3 current_pos_offset;
	for (const auto neighbor_cell_index : possible_neighbor_indices)
	{
		if (neighbor_cell_index == -1)
		{
			continue;
		}

		// - Access each boid in the cell and compute velocity change from
		//   the boids rules, if this boid is within the neighborhood distance.
		for (int iother = gridCellStartIndices[neighbor_cell_index]
			; iother < gridCellEndIndices[neighbor_cell_index]
			; iother++)
		{
			if (iother == index){ continue; }

			current_pos_offset = pos[iother] - pos[index];
			current_distance = glm::length(current_pos_offset);

			// Rule 1: Get neighbor position sum and neighbor count for rule1
			if (current_distance < rule1Distance)
			{
				rule1_neighbor_pos_sum += pos[iother];
				rule1_neighbor_count += 1.0f;
			}
			// Rule 2: Calculate offset for rule 2
			if (current_distance < rule2Distance)
			{
				rule2_total_offset -= current_pos_offset;
			}
			// Rule 3: Get velocity sum and neighbor count for rule 3
			if (current_distance < rule3Distance)
			{
				rule3_neighbor_vel_sum += vel1[iother];
				rule3_neighbor_count += 1.0f;
			}
		}
	}

	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	if (rule1_neighbor_count > 0.0f)
	{
		delta_vel += rule1Scale * ((rule1_neighbor_pos_sum / rule1_neighbor_count) - pos[index]);
	}

	// Rule 2: boids try to stay a distance d away from each other
	delta_vel += rule2Scale *  rule2_total_offset;

	// Rule 3: boids try to match the speed of surrounding boids
	if (rule3_neighbor_count > 0.0f)
	{
		delta_vel += rule3Scale * ((rule3_neighbor_vel_sum / rule3_neighbor_count) - vel1[index]);
	}

	auto new_vel = vel1[index] + delta_vel;

	// - Clamp the speed change before putting the new speed in vel2
	auto speed = glm::length(new_vel);
	if (speed > maxSpeed)
	{
		new_vel = new_vel * (maxSpeed / speed);
	}
	vel2[index] = new_vel;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt)
{

    // DONE-1.2 - use the kernels you wrote to step the simulation forward in time.
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize >>>(numObjects, dev_pos, dev_vel1, dev_vel2);
    kernUpdatePos <<<fullBlocksPerGrid, blockSize >>>(numObjects, dt, dev_pos, dev_vel1);

    // DONE-1.2 ping-pong the velocity buffers
    // swap pointers so current velocity become new velocity
    std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt)
{
    dim3 fullBlocksPerGridForBoids((numObjects + blockSize - 1) / blockSize);
    dim3 fullBlocksPerGridForCells((gridCellCount + blockSize - 1) / blockSize);
    // DONE-2.1
    // Uniform Grid Neighbor search using Thrust sort.
    // In Parallel:

    // - label each particle with its array index as well as its grid index.
    //   Use 2x width grids.
	kernComputeIndices << <fullBlocksPerGridForBoids, blockSize >> >(numObjects, gridSideCount
        , gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    
    // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
    //   are welcome to do a performance comparison.
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects
        , dev_thrust_particleArrayIndices);

    //   reset start and end buffer since not all values are set during kernIdentifyCellStartEnd
    kernResetIntBuffer <<<fullBlocksPerGridForCells, blockSize >>>(gridCellCount
        , dev_gridCellStartIndices, -1);
    kernResetIntBuffer <<<fullBlocksPerGridForCells, blockSize >>>(gridCellCount
        , dev_gridCellEndIndices, -1);

    // - Naively unroll the loop for finding the start and end indices of each
    //   cell's data pointers in the array of boid indices
    kernIdentifyCellStartEnd <<<fullBlocksPerGridForBoids, blockSize >>>(numObjects
        , dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

    // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchScattered <<<fullBlocksPerGridForBoids, blockSize >>>(numObjects
        , gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices
        , dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

    // - Update positions
    kernUpdatePos <<<fullBlocksPerGridForBoids, blockSize >>>(numObjects, dt, dev_pos, dev_vel1);

    // - Ping-pong buffers as needed
    std::swap(dev_vel1, dev_vel2);
}



void Boids::stepSimulationCoherentGrid(float dt) {
	dim3 fullBlocksPerGridForBoids((numObjects + blockSize - 1) / blockSize);
	dim3 fullBlocksPerGridForCells((gridCellCount + blockSize - 1) / blockSize);
	// DONE-2.3 - start by copying Boids::stepSimulationNaiveGrid
	// Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
	// In Parallel:

	// - Label each particle with its array index as well as its grid index.
	//   Use 2x width grids
	kernComputeIndices <<<fullBlocksPerGridForBoids, blockSize >>>(numObjects, gridSideCount
		, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects
		, dev_thrust_particleArrayIndices);

	//   reset start and end buffer since not all values are set during kernIdentifyCellStartEnd
	kernResetIntBuffer <<<fullBlocksPerGridForCells, blockSize >>>(gridCellCount
		, dev_gridCellStartIndices, -1);
	kernResetIntBuffer <<<fullBlocksPerGridForCells, blockSize >>>(gridCellCount
		, dev_gridCellEndIndices, -1);

	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	kernIdentifyCellStartEnd <<<fullBlocksPerGridForBoids, blockSize >>>(numObjects
		, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

	// - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
	//   the particle data in the simulation array.
	//   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
	kernReshuffle <<<fullBlocksPerGridForBoids, blockSize >>>(numObjects, dev_pos
		, dev_vel1, dev_pos_shuffler, dev_vel_shuffler, dev_particleArrayIndices);
	std::swap(dev_pos, dev_pos_shuffler);
	std::swap(dev_vel1, dev_vel_shuffler);

	// - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchCoherent <<<fullBlocksPerGridForBoids, blockSize >>>(numObjects
		, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices
		, dev_gridCellEndIndices, dev_pos, dev_vel1, dev_vel2);

	// - Update positions
	kernUpdatePos <<<fullBlocksPerGridForBoids, blockSize >>>(numObjects, dt, dev_pos, dev_vel1);

	// - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
	std::swap(dev_vel1, dev_vel2);
}

void Boids::endSimulation() 
{
	cudaFree(dev_vel1);
	cudaFree(dev_vel2);
	cudaFree(dev_pos);

	// DONE-2.1 DONE-2.3 - Free any additional buffers here.
	cudaFree(dev_particleArrayIndices);
	cudaFree(dev_particleGridIndices);
	cudaFree(dev_gridCellStartIndices);
	cudaFree(dev_gridCellEndIndices);
	cudaFree(dev_pos_shuffler);
	cudaFree(dev_vel_shuffler);
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
