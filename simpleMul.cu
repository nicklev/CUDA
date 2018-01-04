#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>

#define TILE_WIDTH 16

#define cudaCheckError() {                               \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                                \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                      \
    exit(1);                                             \
  }                                                      \
}

__global__ void SimpleMulKernel (float *Nd, float *Pd, int width, int height)
{
	int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
	
	if (Row < height && Col < width)
	{
		float Pvalue = 0;
	
		for (int k = 0; k < height; k++)
			Pvalue += Nd[k*width+Row] * Nd[k*width+Col];
	
		Pd[Row*width+Col] = Pvalue;
	}
}

int main(int argc, char* argv[])
{
	float *A_h, *C_h;
	float *A_d, *C_d;
	int i, width, height, size_A, size_C;

	cudaEvent_t start, stop;
  	cudaEventCreate(&start);
  	cudaEventCreate(&stop);
		
	srand(time(NULL));
	
	if (argc != 3)
	{
		printf("Provide the problem size.\n");
		return -1;
	}
	
	width = atoi(argv[2]);
	height = atoi(argv[1]);
	
	size_A = width * height * sizeof(float);
	size_C = height * height * sizeof(float);
	
	//memory allocation for host matrixes
	A_h = (float *)malloc(size_A);
	C_h = (float *)malloc(size_C);
	
	if ((A_h == NULL) || (C_h == NULL))
	{
		printf("Could not allocate memory.\n");
		return -2;
	}
	
	//initialization of matrixes
	for (i = 0; i < width*height; i++) {
		A_h[i] = (rand() % 100) / 100.00;
	}
	
	//memory allocation of device matrixes
	cudaMalloc((void**) &A_d, size_A); cudaCheckError();
	cudaMalloc((void**) &C_d, size_C); cudaCheckError();
	
	//copy Host matrixes to Device matrixes
	cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice); cudaCheckError();
	
	//dimensions of device
	dim3 dimGrid(((width-1)/TILE_WIDTH)+1, ((height-1)/TILE_WIDTH)+1, 1);
	dim3 dimBLock(TILE_WIDTH,TILE_WIDTH,1);
	
	cudaEventRecord(start);
	//calculation of multiplication
	SimpleMulKernel<<<dimGrid, dimBLock>>>(A_d, C_d, width, height);
	cudaCheckError();
	cudaEventRecord(stop);

	//copy device results to host
	cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost); cudaCheckError();
	
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Milliseconds: %f\n", milliseconds);

	//free device memory
	cudaFree(A_d); cudaCheckError();
	cudaFree(C_d); cudaCheckError();
}