// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "CycleTimer.h"
#define BLOCK_SIZE 16
#define CWIDTH 32
#define CHEIGHT 32
#define FWIDTH 3
#define FHEIGHT 3
#define CNUMBER 32

//__shared__ float filteroutshared[32][32];

__global__ void convolute_kernel(float *IC, float*F, float *OC) {
    //printf("hello");

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
			c_acc[i] += A[(idy*8 + idx/4) + j*256] * B[idx%4 + 4*i +256*j]; 	
		}
	}

    for(int i = 0; i < k/4; i++)
    {
        C[(idy*8 + idx/4)*256 + idx%4 + 4*i] = c_acc[i];
    }
    */
    /*
    int kCenterX = FWIDTH / 2;
    int kCenterY = FWIDTH / 2;
    int a[2];
    a[0] = 0;
    a[1] = 0;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    if(i < CWIDTH && j < CWIDTH)
    {
        for(int k = 0; k < CNUMBER; k++)
        {
        for (int m = 0; m < FWIDTH; ++m)     // kernel rows
        {

                for (int n = 0; n < FWIDTH; ++n) // kernel columns
                {

                    // index of input, used for checking boundary
                    int ii = i + (m - kCenterY);
                    int jj = j + (n - kCenterX);

                    // ignore input samples which are out of bound
                    if (ii >= 0 && ii < CWIDTH && jj >= 0 && jj < CWIDTH)
                        a[k] += IC[k * CWIDTH * CWIDTH + ii * CWIDTH + jj] * F[k * FWIDTH * FWIDTH + m * FWIDTH + n];
                    }
                }
                OC[k + CWIDTH * CWIDTH + i * CWIDTH + j] = a[k];
        }

    }
    */
    /*
    int idx = threadIdx.x;
    int idy = threadIdx.y;
    int kCenterX = FWIDTH / 2;
    int kCenterY = FWIDTH / 2;
    int octemp[3][64];
    __shared__ int filtemp[CNUMBER * FWIDTH * FWIDTH];

    int a,b;
    for(int j = 0; j < CNUMBER; j++)
    {
        for(int i = 0; i < 64; i++)
            octemp[j][i] = 0;
    }

    if(idx < CNUMBER * FWIDTH * FWIDTH && idy == 0)
        filtemp[idx] = F[idx];
    
    __syncthreads();

    for(int i = 0; i < CWIDTH/4; i++)
    {
        a = (idy*8 + idx/4);
        b = idx%4 + 4*i;
        for (int m = 0; m < FWIDTH; ++m)     // kernel rows
        {

            for (int n = 0; n < FWIDTH; ++n) // kernel columns
            {

                // index of input, used for checking boundary
                int ii = a + (m - kCenterY);
                int jj = b + (n - kCenterX);
                for(int k = 0; k < CNUMBER; k++)
                {
                // ignore input samples which are out of bound
                    if (ii >= 0 && ii < CWIDTH && jj >= 0 && jj < CWIDTH)
                    octemp[k][i] += IC[k * CWIDTH * CWIDTH + ii * CWIDTH + jj] * filtemp[k * FWIDTH * FWIDTH + m * FWIDTH + n];
                }
            }
        
        }
    }
    for(int j = 0; j < CNUMBER; j++)
    {
        for(int i = 0; i < CWIDTH/4; i++)
        {

            OC[(j*CWIDTH*CWIDTH) + ((idy*8 + idx/4)*CWIDTH) + idx%4 + 4*i] = octemp[j][i];

        }
    }
    */



    int idx = threadIdx.x;
    int idy = threadIdx.y;

    //printf("hello\n");
    int kCenterX = FWIDTH / 2;
    int kCenterY = FHEIGHT / 2;
    int filtersperthread = CNUMBER / BLOCK_SIZE;
    int remaining = CNUMBER % BLOCK_SIZE;
    //printf("fil per t %d",filtersperthread);
    float filtertemp[3][FHEIGHT][FWIDTH];
    float outputtemp[CHEIGHT / BLOCK_SIZE + 1][CWIDTH];
    //printf("hello");

    for(int k = 0; k < filtersperthread; k++)
    {

        for(int i = 0; i < FHEIGHT; i++)
        {
            for(int j = 0; j < FWIDTH; j++)
            {

                filtertemp[k][i][j] = F[k * FHEIGHT * FWIDTH + i * FWIDTH + j];
                //printf("%.1f    %.1f\n",filtertemp[k][i][j],F[k * FHEIGHT * FWIDTH * BLOCK_SIZE + idx%BLOCK_SIZE * FHEIGHT * FWIDTH + i * FWIDTH + j]); 

            }
        }
    } 
    /*    
    if(idx == 0 && idy == 0)
    {
        for(int i = 0; i < FHEIGHT; i++)
        {
            for(int j = 0; j < FWIDTH; j++)
            {
                printf("%.1f",filtertemp[0][i][j]);
            }
            printf("\n");
        }
    }
    */
    
    if(remaining > 0 && idx < remaining)
    {

        int k =  filtersperthread;  
        for(int i = 0; i < FHEIGHT; i++)
        {
            for(int j = 0; j < FWIDTH; j++)
            {
                filtertemp[k][i][j] = F[k * FHEIGHT * FWIDTH + i * FWIDTH + j];
            }     
        }  
    }
        //idx = column number, idy = row number
    
    for(int k = 0; k < filtersperthread; k++)
    {

        for(int i = (idy * CHEIGHT / BLOCK_SIZE) - kCenterY; i < ((idy + 1) * CHEIGHT / BLOCK_SIZE) + kCenterY; i++)
        {
            for(int j = 0; j < CWIDTH; j++)
            {
                if(i >= 0 && i < CHEIGHT)
                {
                    int input_element = IC[(k * CHEIGHT * CWIDTH * BLOCK_SIZE) + ((idx%BLOCK_SIZE) * CHEIGHT * CWIDTH) + (i * CWIDTH) + j];
                    #pragma unroll
                    for (int m = 0; m < FHEIGHT; ++m)     // kernel rows
                    {

                        for (int n = 0; n < FWIDTH; ++n) // kernel columns
                        {
                            int ii = i + (m - kCenterY);
                            int jj = j + (n - kCenterX);
                            if (ii >= 0 && ii < CHEIGHT && jj >= 0 && jj < CWIDTH && ii >= (idy * CHEIGHT / BLOCK_SIZE) && ii < ((idy + 1) * CHEIGHT / BLOCK_SIZE))
                            {
                                //printf("%.1f\n", filtertemp[k][m][n] * IC[idx * CWIDTH * CWIDTH + i * CWIDTH + j]);
                                 outputtemp[ii - (idy * CHEIGHT / BLOCK_SIZE)][jj] += filtertemp[k][m][n] * input_element;
                            }
                                //__syncthreads();
                        }
                    }
                }
            }
        }
        for(int i = (idy * CHEIGHT / BLOCK_SIZE); i < ((idy + 1) * CHEIGHT / BLOCK_SIZE); i++)
        {
            for(int j = 0; j < CWIDTH; j++)
            {
                OC[(k * CHEIGHT * CWIDTH * BLOCK_SIZE) + ((idx%BLOCK_SIZE) * CHEIGHT * CWIDTH) + (i * CWIDTH) + j] = outputtemp[i - (idy * CHEIGHT / BLOCK_SIZE)][j];
                outputtemp[i - (idy * CHEIGHT / BLOCK_SIZE)][j] = 0;
            }
        }
    }
    
    if(remaining > 0 && idx < remaining)
    {

        int k =  filtersperthread;  
        for(int i = (idy * CHEIGHT / BLOCK_SIZE) - kCenterY; i < ((idy + 1) * CHEIGHT / BLOCK_SIZE) + kCenterY; i++)
        {
            for(int j = 0; j < CWIDTH; j++)
            {
                if(i >= 0 && i < CHEIGHT)
                {
                    int input_element = IC[k * CHEIGHT * CWIDTH * BLOCK_SIZE + idx%BLOCK_SIZE * CHEIGHT * CWIDTH + i * CWIDTH + j];
                    #pragma unroll
                    for (int m = 0; m < FHEIGHT; ++m)     // kernel rows
                    {

                        for (int n = 0; n < FWIDTH; ++n) // kernel columns
                        {
                            int ii = i + (m - kCenterY);
                            int jj = j + (n - kCenterX);
                            //if(idx == 0 && idy == 0)
                            //printf(" %d %d %d %d ", m, n, ii, jj);
                            if (ii >= 0 && ii < CHEIGHT && jj >= 0 && jj < CWIDTH && ii >= (idy * CHEIGHT / BLOCK_SIZE) && ii < ((idy + 1) * CHEIGHT / BLOCK_SIZE))
                            {
                                //printf("%f", filtertemp[k][m][n] * IC[idx * CWIDTH * CWIDTH + i * CWIDTH + j]);
                                 outputtemp[ii - (idy * CHEIGHT / BLOCK_SIZE)][jj] += filtertemp[k][m][n] * input_element;
                                // if(idx == 0 && idy == 0)
                                 //printf(" yo %d %d \n", m, n);
                            }
                                //__syncthreads();
                        }
                    }
                }
            }
        }
        for(int i = (idy * CHEIGHT / BLOCK_SIZE); i < ((idy + 1) * CHEIGHT / BLOCK_SIZE); i++)
        {
            for(int j = 0; j < CWIDTH; j++)
            {
                OC[k * CHEIGHT * CWIDTH * BLOCK_SIZE + (idx % BLOCK_SIZE) * CHEIGHT * CWIDTH + i * CWIDTH + j] = outputtemp[i - (idy * CHEIGHT / BLOCK_SIZE)][j];
                outputtemp[i - (idy * CHEIGHT / BLOCK_SIZE)][j] = 0;
            }
        }
    }
    
    /*    
    for(int i = 0; i < 32; i++)
        filteroutshared[idy][idx] = 0;


    for(int i = 0; i < 32; i++)
        filteroutshared[idy][(idx + i) % 32] += outputtemp[0][(idx + i) % 32];

    __syncthreads();
    if(idx == 0 && idy == 0)
    {
        for(int i = 0; i < 32; i++)
        {
            for(int j = 0; j < 32; j++)
                printf("%.1f ", filteroutshared[i][j]);
            printf("\n");
        }
        
            
    }
    
    */

}

/**
 * Run a simple test of matrix multiplication using CUDA
 */

void cudaconvolute(float* IC, float* F, float* OC, float*** OC_cpu)
{
    float totalBytes_channel = sizeof(float) *  CNUMBER * CHEIGHT * CWIDTH; 
    float totalBytes_filter =  sizeof(float) *  CNUMBER * FHEIGHT * FWIDTH;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
   // dim3 blocks(N/BLOCK_SIZE,N/BLOCK_SIZE);


    float* device_IC;
    float* device_F;
    float* device_OC;

    cudaMalloc(&device_IC, totalBytes_channel);
    cudaMalloc(&device_F, totalBytes_filter);
    cudaMalloc(&device_OC, totalBytes_channel);

    cudaMemcpy(device_IC, IC, totalBytes_channel, cudaMemcpyHostToDevice);
    cudaMemcpy(device_F, F, totalBytes_filter, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double startKernelTime = CycleTimer::currentSeconds();


    cudaEventRecord(start);
    convolute_kernel<<<1, threadsPerBlock>>>(device_IC, device_F, device_OC);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    double endKernelTime = CycleTimer::currentSeconds();


    cudaMemcpy(OC, device_OC, totalBytes_channel, cudaMemcpyDeviceToHost);
    

    double kernelDuration = endKernelTime - startKernelTime;
    printf("KernelDuration: %.3f ms\n", 1000.f * kernelDuration);

    float m = 0;
    cudaEventElapsedTime(&m, start, stop);
    printf("CUDA Elapsed Time %f ms\n", m);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("Bandwidth = %f GFLOPS/s\n", (2 * FWIDTH * FHEIGHT * CWIDTH * CHEIGHT * CNUMBER) / (m * 1000000000));

   // for(int k = 0; k < CNUMBER; k++) {
     //   for (int i = 0;i < CHEIGHT; i++){
       //     for (int j = 0; j < CWIDTH; j++) {


    bool equal = true;
    for(int k = 0; k < CNUMBER; k++) {
        for (int i = 0;i < CHEIGHT; i++){
            for (int j = 0; j < CWIDTH; j++) {
                //printf("%0.1f ",OC[k * CHEIGHT * CWIDTH + i * CWIDTH + j]);
                if(OC_cpu[k][i][j] != OC[k * CHEIGHT * CWIDTH + i * CWIDTH + j])
                {
                    equal = false;
                    //printf("%0.1f ",OC[k * CHEIGHT * CWIDTH + i * CWIDTH + j]);
                        //printf("%d %d %d %.1f != %.1f\n", k , i, j, OC_cpu[k][i][j], OC[k * CHEIGHT * CWIDTH + i * CWIDTH + j]);
                        //break;
                }
                //printf("%d",OC[k * CNUMBER * CWIDTH + i * CWIDTH + j]);      
            }
            //printf("\n");
        }
        //printf("\n");
    }
    if(equal)
	printf("EQUAL\n");
    else
    printf("NOT EQUAL\n");
    cudaFree(device_IC);
    cudaFree(device_F);
    cudaFree(device_OC);
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
        printf("   Shared memory per block:   %d bytes\n", deviceProps.sharedMemPerBlock);
    }
    printf("---------------------------------------------------------\n");
}

//cudaEvent_t start, stop;
 // cudaEventCreate(&start);
 // cudaEventCreate(&stop);
 //cudaEventRecord(start);
 //cudaEventRecord(stop);
 //float m = 0;
  //cudaEventElapsedTime(&m, start, stop);

  //cudaEventDestroy(start);
  //cudaEventDestroy(stop);