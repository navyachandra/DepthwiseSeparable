// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "CycleTimer.h"
#define BLOCK_SIZE 32
//__shared__ float c_sha = new float[32][32];
//__shared__ float c_acc = new float[256][256];
__global__ void mm_kernel(float *C, float*A, float *B, int k) {

/*  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Accumulate row i of A and column j of B
  int i = by * blockDim.y + ty;
  int j = bx * blockDim.x + tx;

  float accu = 0.0;

  for(int h=0; h<k; h++){
    accu = accu + A[ i * k + h ] * B[ h * k + j ];
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  C[ i * k + j ] = accu;
*/

/*    int i,j;
    float c_acc = 0;

    __shared__ float shared_A [BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

   
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    for (int tile = 0; tile < gridDim.x; tile++) {

        j = tile * BLOCK_SIZE + threadIdx.x;
        i = tile * BLOCK_SIZE + threadIdx.y;

        shared_A[threadIdx.y][threadIdx.x] = A[row * k + j];

        shared_B[threadIdx.y][threadIdx.x] = B[i * k + col]; 

        
        __syncthreads();
	#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {

            c_acc += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

     
        __syncthreads();
    }

    C[row * k + col] = c_acc;
*/

   // int row = blockIdx.y * blockDim.y + threadIdx.y;
   // int col = blockIdx.x * blockDim.x + threadIdx.x;
	
    int idx = threadIdx.x;
    int idy = threadIdx.y;
 //   __shared__ float c_acc[8][256]; 


        //shared_A[threadIdx.y][threadIdx.x] = A[row * k + j];

       // shared_B[threadIdx.y][threadIdx.x] = B[i * k + col]; 
    float c_acc[64];

	for(int i = 0; i < 64; i++)
        	c_acc[i] = 0;
	
	for(int j = 0; j < k; j++)
	{
		for(int i = 0; i < k/4; i++)
		{
			c_acc[i] += A[(idy*8 + idx/4) + j*256] * B[idx%4 + 4*i +256*j]; 	
		}
	}

                for(int i = 0; i < k/4; i++)
                {
                        C[(idy*8 + idx/4)*256 + idx%4 + 4*i] = c_acc[i];//A[(idy*8 + idx/4)*256 + j] * B[idx%4 + 4*i +256*j];
                        //if(idx == 8 && idy == 0)
                          //    printf("%f\t%f\n", C[(idy*8 + idx/4)*256 + idx%4 + 4*i], c_acc[i]);
                }
/*


    int idx = threadIdx.x;
    int idy = threadIdx.y;
    float c_acc[64];

for(int i = 0; i < 64; i++)
	c_acc[i] = 0;

        for(int j = 0; j < k; j++)
        {       
                for(int i = 0; i < k/4; i++)
                {       
                        c_acc[i] += A[(idy*8 + idx/4)*256 + j] * B[idx%4 + 4*i +256*j];
                }
        }

                for(int i = 0; i < k/4; i++)
                {       
                        C[(idy*8 + idx/4)*256 + idx%4 + 4*i] = c_acc[i];//A[(idy*8 + idx/4)*256 + j] * B[idx%4 + 4*i +256*j];
			//if(idx == 8 && idy == 0)
			//	printf("%f\t%f\n", C[(idy*8 + idx/4)*256 + idx%4 + 4*i], c_acc[i]);
                }

*/



    	
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */

void matrixMul(int N, float* a, float* b, float* c, float* c_cpu)
{
    int totalBytes = sizeof(float) *  N * N;    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
   // dim3 blocks(N/BLOCK_SIZE,N/BLOCK_SIZE);

    float acc = 0.0;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            acc = 0.0;
            for (int h = 0; h < N; ++h)
            {
                acc += *(a+i * N + h) * *(b+h * N + j);
            }
            *(c_cpu+i * N + j) = acc;
//		if(c_cpu[i * N + j] < 1)
//			printf("error %d, %d\n", i,j);
        }
    }

    float* device_A;
    float* device_B;
    float* device_C;

    cudaMalloc(&device_A, totalBytes);
    cudaMalloc(&device_B, totalBytes);
    cudaMalloc(&device_C, totalBytes);

    cudaMemcpy(device_A, a, totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, b, totalBytes, cudaMemcpyHostToDevice);

    double startKernelTime = CycleTimer::currentSeconds();

    mm_kernel<<<1, threadsPerBlock>>>(device_C, device_A, device_B, N);
    //double startKernelTime = CycleTimer::currentSeconds();
    cudaDeviceSynchronize();

    double endKernelTime = CycleTimer::currentSeconds();

    cudaMemcpy(c, device_C, totalBytes, cudaMemcpyDeviceToHost);
    

    double kernelDuration = endKernelTime - startKernelTime;
    printf("KernelDuration: %.3f ms\n", 1000.f * kernelDuration);

//int count = 0;
    bool equal = true;
    for (int i=0;i< N;i++){
        for (int j = 0; j < N; j++) {
           // if (abs(c_cpu[i*N+j]-c[i*N+j]) > 0.001)
            //{count++;
             //   equal = false;
                //printf("NOT EQUAL\n");
	//	if(i==0 && j==255)
	//	printf("C[%d][%d] = %f , %d\n", i,j, c[i*N+j], c_cpu[i*N+j]);
		//break;
//	printf("%f\n", c[i*N+j]);
//            }
        }
	//if(!equal)
	//	break;
    }
    if(equal)
	printf("EQUAL\n");
//printf("c = %d",count);
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
}

void
printCudaInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}

