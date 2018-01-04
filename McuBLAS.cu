#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cublas.h>

#define cudaCheckError() {                               \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                                \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                      \
    exit(1);                                             \
  }                                                      \
}

int main(int argc, char* argv[])
{
  int       height = atoi(argv[1]);
  int		width = atoi(argv[2]);
  int		i,j;
  int       status;

  double    *psa, *psc; //host matrices
  double    *sap, *scp;
  double    *pda, *pdc; //device matrices

  double alpha   = 1.0;
  double beta    = 0.0;
 
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  srand (time(NULL));

  //initializing host and device matrices
  pda = NULL;
  pdc = NULL;
  psa = (double *) malloc(height * width * sizeof(*psa) );
  psc = (double *) malloc(width * width * sizeof(*psc) );

  /* Initialize CUDA */
  printf("Initializing CUDA...");
  status = cublasInit();
  if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! CUBLAS initialization error\n");
      return EXIT_FAILURE;
  }
  printf("Done.\n");

  /* Re-initialize the matrices */
  //clock_gettime(CLOCK_MONOTONIC, &t1);
  sap = psa;
  scp = psc;
  for (i=0; i < width*width; i++)
	  *scp++ = 0.0;
	  
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      *sap++ = (double)rand() / (double)RAND_MAX;
    }
  }

  fflush(stdout);

  /* Allocate device memory for the matrices */
  printf("Starting CUDA DGEMM...");

  status = cublasAlloc(height*width, sizeof(*pda), (void**) &pda);
  if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return EXIT_FAILURE;
  }

  status = cublasAlloc(width*width, sizeof(*pdc), (void**) &pdc);
  if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (C)\n");
      return EXIT_FAILURE;
  }

  /* Initialize the device matrices with the host matrices */
  status = cublasSetVector(height*width, sizeof(*psa), psa, 1, pda, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device access error (write A)\n");
      return EXIT_FAILURE;
  }

  status = cublasSetVector(width*width, sizeof(*psc), psc, 1, pdc, 1);
  //cudaCheckError();
  if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device access error (write C)\n");
      return EXIT_FAILURE;
  }

  /* Clear last error */
  cublasGetError();

  // Create a handle for CUBLAS
  // cublasHandle_t handle;
  // cublasCreate(&handle);

  /* Performs operation using cublas */
  cudaEventRecord(start);
  cublasDgemm('n', 't', width, width, height, alpha, pda, width, pda, width, beta, pdc, width);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  
  // Destroy the handle
  // cublasDestroy(handle);
  
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Milliseconds: %f\n", milliseconds);

  status = cublasGetError();
  if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! kernel execution error.\n");
      return EXIT_FAILURE;
  }

  /* Read the result back */
  status = cublasGetVector(width*width, sizeof(*psc), pdc, 1, psc, 1);
  cudaCheckError();
  if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device access error (read C)\n");
      return EXIT_FAILURE;
  }
}

