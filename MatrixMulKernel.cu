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


__global__ void MatrixMulKernel (float* Nd, float* Pd, int width, int height)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
	
	float Pvalue = 0;
	float tempM, tempN;
	
	if ((0*TILE_WIDTH + ty) < height && Row < width)
		tempM = Nd[(0*TILE_WIDTH + ty)*width + Col];
	else
		tempM = 0.0;
	
	if ((0*TILE_WIDTH + tx) < height && Col < width)
		tempN = Nd[(0*TILE_WIDTH + tx)*width + Row];
	else
		tempN = 0.0;

	for (int m=1; m <= (TILE_WIDTH + height - 1)/TILE_WIDTH; ++m)
	{
		Mds[ty][tx] = tempM;
		Nds[tx][ty] = tempN;
		__syncthreads();
		
		if ((m*TILE_WIDTH + ty) < height && Row < width)
			tempM = Nd[(m*TILE_WIDTH + ty)*width + Col];
		else
			tempM = 0.0;		
		
		if ((m*TILE_WIDTH + tx) < height && Col < width)
			tempN = Nd[(m*TILE_WIDTH + tx)*width + Row];
		else
			tempN = 0.0;

		for (int k=0; k<TILE_WIDTH; ++k)
			Pvalue+=Mds[k][ty] * Nds[k][tx];
		__syncthreads();
	}
	Pd[Row*width + Col] = Pvalue;
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
	
	height = atoi(argv[1]);
	width = atoi(argv[2]);
	
	size_A = width * height * sizeof(float);
	size_C = width* width * sizeof(float);
	
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
	dim3 dimGrid(((width-1)/TILE_WIDTH)+1, ((width-1)/TILE_WIDTH)+1, 1);
	dim3 dimBLock(TILE_WIDTH,TILE_WIDTH,1);
	
	cudaEventRecord(start);
	//calculation of multiplication
	MatrixMulKernel<<<dimGrid, dimBLock>>>(A_d, C_d, width, height);
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
	
	//print results
	// for (i = 0; i<width*height; i++)
	// {
		// if(i % width == 0)
		// {
			// printf("\n");
		// }
		// printf("%f, ", A_h[i]);
	// }
	// printf("\n\n");
	// printf("\n");
		
	// for (i = 0; i<width*width; i++)
	// {
		// if(i % width == 0)
		// {
			// printf("\n");
		// }
		// printf("%f, ", C_h[i]);
	// }
	// printf("\n\n");
	// printf("\n");
}