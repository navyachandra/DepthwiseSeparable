// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "CycleTimer.h"
#define BLOCK_SIZE 16
#define CWIDTH 16
#define CHEIGHT 16
#define FWIDTH 3
#define FHEIGHT 3
#define CNUMBER 16
#define COUT 16

__shared__ float filteroutshared[CNUMBER][CNUMBER];

__global__ void convolute_kernel(float *IC, float*F, float*FP, float *SOC) {


    int idx = threadIdx.x;
    int idy = threadIdx.y;

    //printf("hello\n");
    int kCenterX = FWIDTH / 2;
    int kCenterY = FHEIGHT / 2;
    int filtersperthread = CNUMBER / BLOCK_SIZE;
    int remaining = CNUMBER % BLOCK_SIZE;

    // TODO : change these when generalizing
    float filtertemp[1][FHEIGHT][FWIDTH];
    float outputtemp[CHEIGHT / BLOCK_SIZE][CWIDTH];
    //printf("hello");

    float sepFilter[CNUMBER];
  
    for(int i = 0 ; i < CNUMBER; i++)
        sepFilter[i] = FP[idx * CNUMBER + i];

    for(int k = 0; k < filtersperthread; k++)
    {

        for(int i = 0; i < FHEIGHT; i++)
        {
            for(int j = 0; j < FWIDTH; j++)
            {

                filtertemp[k][i][j] = F[k * FHEIGHT * FWIDTH * BLOCK_SIZE + idx%BLOCK_SIZE * FHEIGHT * FWIDTH + i * FWIDTH + j];
                //printf("%.1f    %.1f\n",filtertemp[k][i][j],F[k * FHEIGHT * FWIDTH * BLOCK_SIZE + idx%BLOCK_SIZE * FHEIGHT * FWIDTH + i * FWIDTH + j]); 

            }
        }
    } 
   
    if(remaining > 0 && idx < remaining)
    {

        int k =  filtersperthread;  
        for(int i = 0; i < FHEIGHT; i++)
        {
            for(int j = 0; j < FWIDTH; j++)
            {
                filtertemp[k][i][j] = F[k * FHEIGHT * FWIDTH * BLOCK_SIZE + idx%BLOCK_SIZE * FHEIGHT * FWIDTH + i * FWIDTH + j];
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

    //idx = column number, idy = row number
    
    for(int k = 0; k < filtersperthread; k++)
    {

        for(int i = (idy * CHEIGHT / BLOCK_SIZE) - kCenterY; i < ((idy + 1) * CHEIGHT / BLOCK_SIZE) + kCenterY; i++)
        {
            for(int j = 0; j < CWIDTH; j++)
            {
                if(i >= 0 && i < CHEIGHT)
                {
                    int input_element = IC[k * BLOCK_SIZE + i * CNUMBER * CWIDTH + j * CNUMBER + idx];
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
                                //if (ii >= 0 && ii < CHEIGHT && jj >= 0 && jj < CWIDTH && ii >= (idy * CHEIGHT / BLOCK_SIZE) && ii < ((idy + 1) * CHEIGHT / BLOCK_SIZE))
                                //printf("%.f\n",outputtemp[ii - (idy * CHEIGHT / BLOCK_SIZE)][jj]);
                        }
                    }
                }
            }
        }
        /*
        for(int i = (idy * CHEIGHT / BLOCK_SIZE); i < ((idy + 1) * CHEIGHT / BLOCK_SIZE); i++)
        {
            for(int j = 0; j < CWIDTH; j++)
            {
                OC[k * BLOCK_SIZE + i * CNUMBER * CWIDTH + j * CNUMBER + idx] = outputtemp[i - (idy * CHEIGHT / BLOCK_SIZE)][j];
                outputtemp[i - (idy * CHEIGHT / BLOCK_SIZE)][j] = 0;
            }
        }
        */
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
                    int input_element = IC[k * BLOCK_SIZE + i * CNUMBER * CWIDTH + j * CNUMBER + idx];
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
        /*
        for(int i = (idy * CHEIGHT / BLOCK_SIZE); i < ((idy + 1) * CHEIGHT / BLOCK_SIZE); i++)
        {
            for(int j = 0; j < CWIDTH; j++)
            {
                OC[k * BLOCK_SIZE + i * CNUMBER * CWIDTH + j * CNUMBER + idx] = outputtemp[i - (idy * CHEIGHT / BLOCK_SIZE)][j];
                outputtemp[i - (idy * CHEIGHT / BLOCK_SIZE)][j] = 0;
            }
        }
        */
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
    /*
    if(idx == 0 && idy == 5)
    {
        for(int i = (idy * CHEIGHT / BLOCK_SIZE); i < ((idy + 1) * CHEIGHT / BLOCK_SIZE); i++)
        {
            for(int j = 0; j < CWIDTH; j++)
            {
                //if (ii >= 0 && ii < CHEIGHT && jj >= 0 && jj < CWIDTH && ii >= (idy * CHEIGHT / BLOCK_SIZE) && ii < ((idy + 1) * CHEIGHT / BLOCK_SIZE))
                printf("%.f ",outputtemp[i - (idy * CHEIGHT / BLOCK_SIZE)][j]);
            }
            printf("\n");
        }
    }
    */
    
    float pointwisetemp;
    for(int k = 0; k < BLOCK_SIZE; k++)
    {
        if(idy == k)
        {
            for(int j = 0; j < CHEIGHT / BLOCK_SIZE; j++)
                for(int i = 0; i < CWIDTH; i++)
                    filteroutshared[i][idx] = outputtemp[j][i];
        }
        __syncthreads();
        /*    
        if(idx == 0 && idy == 0)
        {
            for(int i = 0; i < 16; i++)
            {
                for(int j = 0; j < 16; j++)
                {
                    printf("%.1f ",filteroutshared[i][j]);
                }
                printf("\n");
            }
        }
        */
        if(idy == k)
        {        
            #pragma unroll
            for(int j = 0; j < COUT; j++)
            {
                pointwisetemp = 0.0;
                for(int i = 0; i < CWIDTH; i++)
                {
                    pointwisetemp += filteroutshared[j][(idx + i) % CNUMBER] * sepFilter[(idx + i) % CNUMBER];
                }
                SOC[idy * COUT * CWIDTH + j * COUT + idx] = pointwisetemp;
                //if(idy == 0 && idx == 1)
                //printf("%.f  ",pointwisetemp);
            }
        }
        __syncthreads();
    }
    


}

/**
 * Run a simple test of matrix multiplication using CUDA
 */

void cudaconvolute(float* IC, float* F, float* FP, float* SOC, float*** OC_cpu, float*** SOC_cpu)
{
    float totalBytes_channel = sizeof(float) *  CNUMBER * CHEIGHT * CWIDTH; 
    float totalBytes_filter =  sizeof(float) *  CNUMBER * FHEIGHT * FWIDTH;
    float totalBytes_output = sizeof(float) *  COUT * CHEIGHT * CWIDTH; 
    float totalBytes_sepfilter =  sizeof(float) *  CNUMBER * COUT; 

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
   // dim3 blocks(N/BLOCK_SIZE,N/BLOCK_SIZE);

    float* device_IC;
    float* device_F;
    float* device_SOC;
    float* device_FP;

    cudaMalloc(&device_IC, totalBytes_channel);
    cudaMalloc(&device_F, totalBytes_filter);
    cudaMalloc(&device_SOC, totalBytes_output);
    cudaMalloc(&device_FP, totalBytes_sepfilter);

    cudaMemcpy(device_IC, IC, totalBytes_channel, cudaMemcpyHostToDevice);
    cudaMemcpy(device_F, F, totalBytes_filter, cudaMemcpyHostToDevice);
    cudaMemcpy(device_FP, FP, totalBytes_sepfilter, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double startKernelTime = CycleTimer::currentSeconds();


    cudaEventRecord(start);
    convolute_kernel<<<1, threadsPerBlock>>>(device_IC, device_F, device_FP, device_SOC);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    double endKernelTime = CycleTimer::currentSeconds();


    cudaMemcpy(SOC, device_SOC, totalBytes_output, cudaMemcpyDeviceToHost);
    

    double kernelDuration = endKernelTime - startKernelTime;
    printf("KernelDuration: %.3f ms\n", 1000.f * kernelDuration);

    float m = 0;
    cudaEventElapsedTime(&m, start, stop);
    printf("CUDA Elapsed Time %f ms\n", m);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    



    bool equal = true;

    double fops = 0.0;
    for (int i = 0;i < CHEIGHT; i++){
        for (int j = 0; j < CWIDTH; j++) {
            for(int k = 0; k < COUT; k++) {
                fops += SOC[i * CWIDTH * COUT + j * COUT + k];
                if(SOC_cpu[k][i][j] != SOC[i * CWIDTH * COUT + j * COUT + k])
                {
                    equal = false;

                    //printf("%d %d %d %.1f != %.1f\n", k , i, j, SOC_cpu[k][i][j], SOC[i * CWIDTH * COUT + j * COUT + k]);
                        //break;
                }
                //printf("%d",OC[k * COUT * CWIDTH + i * CWIDTH + j]);      
            }
            //printf("\n");
        }
        //printf("\n");
    }

    printf("Bandwidth = %f GFLOPS/s\n", ((2 * FHEIGHT * FWIDTH * CWIDTH * CHEIGHT * CNUMBER) + (2 * COUT * CHEIGHT * CWIDTH * CNUMBER)) / (m * 1000000));

    if(equal)
	printf("EQUAL\n");
    else
    printf("NOT EQUAL\n");
    cudaFree(device_IC);
    cudaFree(device_F);
    cudaFree(device_SOC);
    cudaFree(device_FP);
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

