#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

__global__
void matrix_accels(vector3* d_accels, double* d_hPos, double* d_mass) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < NUMENTITIES && j < NUMENTITIES) {
    if (i == j) {
        FILL_VECTOR(d_accels[i*NUMENTITIES + j], 0, 0, 0);
    } else {
        vector3 distance;
        for (int k = 0; k < 3; ++k) distance[k] = d_hPos[i*3 + k] - d_hPos[j*3 + k];
        double magnitude_sq = distance[0]*distance[0] + distance[1]*distance[1] + distance[2]*distance[2];
        double magnitude = sqrt(magnitude_sq);
        double accelmag = -1 * GRAV_CONSTANT * d_mass[j] / magnitude_sq;
        FILL_VECTOR(d_accels[i*NUMENTITIES + j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
    }
  }
}

__global__
void matrix_sums(vector3* d_accels, double* d_hPos, double* d_hVel, double* d_mass) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < NUMENTITIES) {
    vector3 accel_sum = {0, 0, 0};
    for (int j = 0; j < NUMENTITIES; ++j) {
      for (int k = 0; k < 3; ++k)
        accel_sum[k] += d_accels[i*NUMENTITIES + j][k];
    }

    for (int k = 0; k < 3; ++k) {
      d_hVel[i*3 + k] += accel_sum[k] * INTERVAL;
      d_hPos[i*3 + k] += d_hVel[i*3 + k] * INTERVAL;
    }
  }
}
//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
/*
	int i,j,k;
	vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];
	//first compute the pairwise accelerations.  Effect is on the first argument.
	for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}
	free(accels);
	free(values);
*/
    vector3 *d_accels;
    double *d_hPos, *d_hVel, *d_mass;

    cudaMalloc((void **)&d_accels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    cudaMalloc((void **)&d_hPos, sizeof(double) * NUMENTITIES * 3);
    cudaMalloc((void **)&d_hVel, sizeof(double) * NUMENTITIES * 3);
    cudaMalloc((void **)&d_mass, sizeof(double) * NUMENTITIES);

    cudaMemcpy(d_hPos, hPos, sizeof(double) * NUMENTITIES * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, sizeof(double) * NUMENTITIES * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NUMENTITIES + threadsPerBlock.x - 1) / threadsPerBlock.x, (NUMENTITIES + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_accels<<<blocksPerGrid, threadsPerBlock>>>(d_accels, d_hPos, d_mass);

    matrix_sums<<<(NUMENTITIES + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock.x>>>(d_accels, d_hPos, d_hVel, d_mass);
    cudaDeviceSynchronize();

    cudaMemcpy(hPos, d_hPos, sizeof(double) * NUMENTITIES * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, sizeof(double) * NUMENTITIES * 3, cudaMemcpyDeviceToHost);

    cudaFree(d_accels);
    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_mass);
}
