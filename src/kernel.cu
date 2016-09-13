#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"
//#include <glm/ext.hpp>
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

//////////////////////////////////

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
glm::vec3 *dev_pos_buff; 
glm::vec3 *dev_vel1_buff;
glm::vec3 *dev_vel2_buff;

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

	// TODO-2.1 TODO-2.3 - Allocate additional buffers here.

	cudaMalloc((void**)&dev_pos_buff, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos_buff failed!");
	cudaMalloc((void**)&dev_vel1_buff, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel1_buff failed!");
	cudaMalloc((void**)&dev_vel2_buff, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel2_buff failed!");

	cudaMalloc((void**)&dev_particleArrayIndices, N*sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc-dev_particleArrayIndices-failed!");
	cudaMalloc((void**)&dev_particleGridIndices, N*sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc-dev_particleGridIndices-failed!");
	cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount*sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc-dev_gridCellStartIndices-failed!");
	cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount*sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc-dev_gridCellEndIndices-failed!");
	dev_thrust_particleArrayIndices=thrust::device_pointer_cast(dev_particleArrayIndices);//boids idx table
	dev_thrust_particleGridIndices=thrust::device_pointer_cast(dev_particleGridIndices);//boids grid idx table



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

	glm::vec3   pvj(0.0f, 0.0f, 0.0f); //perceived center of mass v
	glm::vec3   pcj(0.0f, 0.0f, 0.0f); //perceived center of mass 
	glm::vec3 v2  (0.0f, 0.0f, 0.0f); 
	glm::vec3 v3  (0.0f, 0.0f, 0.0f);  
	//int cnt1, cnt2, cnt3 =0; // THIS IS A BUG!!!!!
	float cnt1=0; float cnt2=0; float cnt3=0;
	for (int i=0; i<N; i++){
		if (i!= iSelf) 
		{
			float d = glm::distance(pos[i],pos[iSelf]);
			// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
			if (d < rule1Distance){
				pcj = pcj  +  pos[i];
				cnt1 ++;
			}
			// Rule 2: boids try to stay a distance d away from each other
			if (d < rule2Distance){
				v2 = v2 - (pos[i]-pos[iSelf]);
			}
			// Rule 3: boids try to match the speed of surrounding boids
			if (d < rule3Distance ){
				pvj = pvj + vel[i];
				cnt3 ++;
			} 
		}
	}   	
	//v3 = (pvj - vel[iSelf]);
	//v3 = (pvj )/cnt3-vel[iSelf];
	v3=pvj;
	//v3 /=8;
	glm::vec3 val(0.0,0.0,0.0);
	if (cnt1!=0){
		val+=(pcj/cnt1 - pos[iSelf]) * rule1Scale;
	}
	val+=v3 * rule3Scale+v2*rule2Scale;
	return  val;


	// return glm::vec3(0.0f, 0.0f, 0.0f);

}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
	glm::vec3 *vel1, glm::vec3 *vel2) {

		//Compute a new velocity based on pos and vel1
		int id= blockIdx.x*blockDim.x + threadIdx.x;
		if (id >= N) return;
		glm::vec3 val(0.0,0.0,0.0);
		val+=computeVelocityChange(N,id, pos, vel1)+vel1[id];
		// Clamp the speed
		if (glm::length(val) > maxSpeed){
			val= glm::normalize(val) * maxSpeed;
		}
		vel2[id]=val;

}

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
//             for(z)? Or some other order? -----cool!
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
		if (index >= N) {
			return;
		}		
		//int xi = std::floor((pos[index].x-gridMin.x)*inverseCellWidth);
		//int yi = std::floor((pos[index].y-gridMin.y)*inverseCellWidth);
		//int zi = std::floor((pos[index].z-gridMin.z)*inverseCellWidth);	
		glm::ivec3 xyzi=(pos[index]-gridMin)*inverseCellWidth;
		int gid = gridIndex3Dto1D(xyzi.x, xyzi.y, xyzi.z, gridResolution);
		gridIndices[index] = gid;// - Label each boid with the index of its grid cell.
		indices[index] = index;//- Set up a parallel array of integer indices as pointers to the actual boid data in pos and vel1/vel2
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
		int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (idx >= N) return;
		if (idx >=1){
			if (particleGridIndices[idx-1] != particleGridIndices[idx]){
				gridCellEndIndices[particleGridIndices[idx-1]]=idx-1;//by default -1
				gridCellStartIndices[particleGridIndices[idx]]=idx;//by default -1
			}
		}
		else {
			gridCellStartIndices[particleGridIndices[idx]]=idx;//by default -1
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
		int idx= threadIdx.x + blockIdx.x*blockDim.x;
		if (idx >= N) return;

		// - Identify which cells may contain neighbors. This isn't always 8.
		int pi=particleArrayIndices[idx];

		glm::ivec3 cid0= (pos[idx] - gridMin)*inverseCellWidth;
		cid0=cid0-1;
		cid0.x=imax(cid0.x,0);
		cid0.y=imax(cid0.y,0);
		cid0.z=imax(cid0.z,0);

		glm::vec3   pvj(0.0f, 0.0f, 0.0f); //perceived center of mass v
		glm::vec3   pcj(0.0f, 0.0f, 0.0f); //perceived center of mass 
		glm::vec3 v2  (0.0f, 0.0f, 0.0f); 
		glm::vec3 v3  (0.0f, 0.0f, 0.0f);  
		float cnt1=0; float cnt2=0; float cnt3=0;
		// - For each cell, read the start/end indices in the boid pointer array.
		// PS: assume sequential (sorted already)
		for (int i=0; i <2 ; i++){
			for (int j=0; j<2 ; j++){
				for (int k=0; k <2; k++){
					//if (i==1&&j==1&&k==1) continue;
					if (cid0.x+i<gridResolution && cid0.y+j<gridResolution && cid0.z+k<gridResolution){
						int gid = gridIndex3Dto1D(cid0.x+i, cid0.y+j, cid0.z+k, gridResolution);
						int gi0=gridCellStartIndices[gid];
						int gix=gridCellEndIndices[gid];
						// - Access each boid in the cell and compute velocity change from
						//   the boids rules, if this boid is within the neighborhood distance.
						for (int bii=gi0; bii <= gix; bii++){

							if (bii!=idx){		
								int pii=particleArrayIndices[bii];
								float d = glm::distance(pos[pii],pos[idx]);
								// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
								if (d < rule1Distance){
									pcj = pcj  +  pos[pii];
									cnt1 ++;
								}
								// Rule 2: boids try to stay a distance d away from each other
								if (d < rule2Distance){
									v2 = v2 - (pos[pii]-pos[idx]);
								}
								// Rule 3: boids try to match the speed of surrounding boids
								if (d < rule3Distance ){
									pvj = pvj + vel1[pii];
									cnt3 ++;
								} 							  
							}
						}

					}
				}	
			}
		}
		v3=pvj;
		//v3 /=8;
		glm::vec3 val(0.0,0.0,0.0);
		if (cnt1!=0){
			val+=(pcj/cnt1 - pos[idx]) * rule1Scale;
		}
		val+=vel1[idx]+v3 * rule3Scale+v2*rule2Scale; 
		//vel2[idx]+=val;
		if (glm::length(val) > maxSpeed){
			val= glm::normalize(val) * maxSpeed;
		}
		vel2[idx]=val;
		// - Clamp the speed change before putting the new speed in vel2
}

__global__ void makeContiguous(int N, glm::vec3 *buff_mess, glm::vec3 *buff_good, int *particleArrayIndices){
	int idx= threadIdx.x + blockIdx.x*blockDim.x;
	if (idx >= N) return;
	buff_good[idx] = buff_mess[particleArrayIndices[idx]];
}
__global__ void undo_makeContiguous(int N, glm::vec3 *buff_mess, glm::vec3 *buff_good, int *particleArrayIndices){
	int idx= threadIdx.x + blockIdx.x*blockDim.x;
	if (idx >= N) return;
	buff_mess[particleArrayIndices[idx]]=buff_good[idx];
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
		// - Identify which cells may contain neighbors. This isn't always 8.
		// - For each cell, read the start/end indices in the boid pointer array.
		//   DIFFERENCE: For best results, consider what order the cells should be
		//   checked in to maximize the memory benefits of reordering the boids data.
		// - Access each boid in the cell and compute velocity change from
		//   the boids rules, if this boid is within the neighborhood distance.
		// - Clamp the speed change before putting the new speed in vel2
		int idx= threadIdx.x + blockIdx.x*blockDim.x;
		if (idx >= N) return;
		// - Identify the grid cell that this particle is in
		/*glm::ivec3 gid3=(pos[idx]-gridMin)*inverseCellWidth;
		int gid = gridIndex3Dto1D(gid3.x,gid3.y, gid3.z, gridResolution);*/
		glm::ivec3 cid0= (pos[idx] - gridMin)*inverseCellWidth;
		cid0=cid0-1;
		cid0.x=imax(cid0.x,0);
		cid0.y=imax(cid0.y,0);
		cid0.z=imax(cid0.z,0);

		glm::vec3   pvj(0.0f, 0.0f, 0.0f); //perceived center of mass v
		glm::vec3   pcj(0.0f, 0.0f, 0.0f); //perceived center of mass 
		glm::vec3 v2  (0.0f, 0.0f, 0.0f); 
		glm::vec3 v3  (0.0f, 0.0f, 0.0f);  
		float cnt1=0; float cnt2=0; float cnt3=0;
		// - For each cell, read the start/end indices in the boid pointer array.
		// PS: assume sequential (sorted already)
		for (int i=0; i <2 ; i++){
			for (int j=0; j<2 ; j++){
				for (int k=0; k <2; k++){
					//if (i==1&&j==1&&k==1) continue;
					if (cid0.x+i<gridResolution && cid0.y+j<gridResolution && cid0.z+k<gridResolution){
						int gid = gridIndex3Dto1D(cid0.x+i, cid0.y+j, cid0.z+k, gridResolution);
						int gi0=gridCellStartIndices[gid];
						int gix=gridCellEndIndices[gid];
						// - Access each boid in the cell and compute velocity change from
						//   the boids rules, if this boid is within the neighborhood distance.
						for (int bii=gi0; bii <= gix; bii++){

							if (bii!=idx){		
								int pii=bii;//assume contiguous data
								float d = glm::distance(pos[pii],pos[idx]);
								// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
								if (d < rule1Distance){
									pcj = pcj  +  pos[pii];
									cnt1 ++;
								}
								// Rule 2: boids try to stay a distance d away from each other
								if (d < rule2Distance){
									v2 = v2 - (pos[pii]-pos[idx]);
								}
								// Rule 3: boids try to match the speed of surrounding boids
								if (d < rule3Distance ){
									pvj = pvj + vel1[pii];
									cnt3 ++;
								} 							  
							}
						}

					}
				}	
			}
		}
		v3=pvj; 
		glm::vec3 val(0.0,0.0,0.0);
		if (cnt1!=0){
			val+=(pcj/cnt1 - pos[idx]) * rule1Scale;
		}
		val+=vel1[idx]+v3 * rule3Scale+v2*rule2Scale;  
		if (glm::length(val) > maxSpeed){
			val= glm::normalize(val) * maxSpeed;
		}
		vel2[idx]=val;
		// - Clamp the speed change before putting the new speed in vel2		

}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
	dim3 N_blocks(std::ceil((float)numObjects/ blockSize)); 
	// TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
	kernUpdateVelocityBruteForce<<<N_blocks, threadsPerBlock>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("stepSimulationNaive-kernUpdateVelocityBruteForce failed!");	
	kernUpdatePos<<<N_blocks,threadsPerBlock>>>(numObjects, dt, dev_pos, dev_vel2); 
	checkCUDAErrorWithLine("stepSimulationNaive-kernUpdatePos failed!");
	// TODO-1.2 ping-pong the velocity buffers
	//std::swap(dev_vel1,dev_vel2);
	dev_vel1=dev_vel2;
}

void Boids::stepSimulationScatteredGrid(float dt) {
	// TODO-2.1
	// Uniform Grid Neighbor search using Thrust sort.
	// In Parallel:
	// - label each particle with its array index as well as its grid index.
	dim3 N_blocks(std::ceil((float)numObjects/ blockSize)); 
	dim3 N_blocks_grids(std::ceil((float) gridCellCount/blockSize));

	//kernComputeIndices(int N, int gridResolution,  glm::vec3 gridMin, float inverseCellWidth,  glm::vec3 *pos, int *indices, int *gridIndices)
	kernComputeIndices <<<N_blocks, threadsPerBlock>>>(numObjects,gridSideCount,gridMinimum,gridInverseCellWidth,dev_pos,dev_particleArrayIndices,dev_particleGridIndices);
	checkCUDAErrorWithLine("kernComputeIndices-failed!");
	//   Use 2x width grids.
	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	// needed for use with thrust
	// LOOK-2.1 Example for using thrust::sort_by_key
	// sort grid idx by boid pos idx
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	//kernResetIntBuffer(int N, int *intBuffer, int value) 
	//kernResetIntBuffer<<<N_blocks, threadsPerBlock>>>(numObjects, dev_gridCellStartIndices, -1); //use -1 indicate not occupied
	kernResetIntBuffer<<<N_blocks_grids, threadsPerBlock>>>(gridCellCount, dev_gridCellStartIndices, -1); //use -1 indicate not occupied
	checkCUDAErrorWithLine("kernResetIntBuffer-dev_gridCellStartIndices-failed!");
	kernResetIntBuffer<<<N_blocks_grids, threadsPerBlock>>>(gridCellCount, dev_gridCellEndIndices, -1); //use -1 indicate not occupied
	checkCUDAErrorWithLine("kernResetIntBuffer-dev_gridCellEndIndices-failed!"); 
	kernIdentifyCellStartEnd<<<N_blocks, threadsPerBlock>>>(numObjects, dev_particleGridIndices,  dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd-failed!");
	// - Perform velocity updates using neighbor search 
	kernUpdateVelNeighborSearchScattered<<<N_blocks, threadsPerBlock>>>(numObjects,gridSideCount,gridMinimum,gridInverseCellWidth,gridCellWidth,dev_gridCellStartIndices,dev_gridCellEndIndices
		,dev_particleArrayIndices,dev_pos,dev_vel1,dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered-failed!");
	// - Update positions
	kernUpdatePos<<<N_blocks, threadsPerBlock>>>(numObjects, dt, dev_pos, dev_vel2);
	// - Ping-pong buffers as needed
	//std::swap(dev_vel1,dev_vel2);
	dev_vel1=dev_vel2;
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

	dim3 N_blocks(std::ceil((float)numObjects/ blockSize)); 
	dim3 N_blocks_grids(std::ceil((float) gridCellCount/blockSize));
	kernComputeIndices <<<N_blocks, threadsPerBlock>>>(numObjects,gridSideCount,gridMinimum,gridInverseCellWidth,dev_pos,dev_particleArrayIndices,dev_particleGridIndices);
	checkCUDAErrorWithLine("kernComputeIndices-failed!");

	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices); 

	kernResetIntBuffer<<<N_blocks_grids, threadsPerBlock>>>(gridCellCount, dev_gridCellStartIndices, -1); //use -1 indicate not occupied
	checkCUDAErrorWithLine("kernResetIntBuffer-dev_gridCellStartIndices-failed!");
	kernResetIntBuffer<<<N_blocks_grids, threadsPerBlock>>>(gridCellCount, dev_gridCellEndIndices, -1); //use -1 indicate not occupied
	checkCUDAErrorWithLine("kernResetIntBuffer-dev_gridCellEndIndices-failed!"); 

	kernIdentifyCellStartEnd<<<N_blocks, threadsPerBlock>>>(numObjects, dev_particleGridIndices,  dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd-failed!"); 

	//dev_pos_buff=dev_pos;
	//dev_vel1_buff=dev_vel1;
	//dev_vel2_buff=dev_vel2;
	makeContiguous<<<N_blocks, threadsPerBlock>>>(numObjects, dev_pos, dev_pos_buff, dev_particleArrayIndices); 
	makeContiguous<<<N_blocks, threadsPerBlock>>>(numObjects, dev_vel1, dev_vel1_buff, dev_particleArrayIndices);  
	makeContiguous<<<N_blocks, threadsPerBlock>>>(numObjects, dev_vel2, dev_vel2_buff, dev_particleArrayIndices);


	//kernUpdateVelNeighborSearchCoherent<<<N_blocks, threadsPerBlock>>>(numObjects,gridSideCount,gridMinimum,gridInverseCellWidth,gridCellWidth,dev_gridCellStartIndices,dev_gridCellEndIndices
	//	,dev_pos,dev_vel1,dev_vel2);
	kernUpdateVelNeighborSearchCoherent <<<N_blocks, threadsPerBlock >> >(numObjects, gridSideCount,
		gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_pos_buff, dev_vel1_buff, dev_vel2_buff);
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered-failed!"); 

	kernUpdatePos <<<N_blocks, threadsPerBlock >>>(numObjects, dt, dev_pos_buff, dev_vel2_buff);

	//dev_vel1=dev_vel2_buff;
	//dev_pos=dev_pos_buff;

	//std::swap(dev_pos, dev_pos_buff);
	//std::swap(dev_vel1, dev_vel2_buff);
	undo_makeContiguous<<<N_blocks, threadsPerBlock>>>(numObjects, dev_pos, dev_pos_buff, dev_particleArrayIndices); 
	undo_makeContiguous<<<N_blocks, threadsPerBlock>>>(numObjects, dev_vel1, dev_vel2_buff, dev_particleArrayIndices);  
	undo_makeContiguous<<<N_blocks, threadsPerBlock>>>(numObjects, dev_vel2, dev_vel2_buff, dev_particleArrayIndices); 
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

	cudaFree(dev_pos_buff);
	cudaFree(dev_vel1_buff);
	cudaFree(dev_vel2_buff);
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
