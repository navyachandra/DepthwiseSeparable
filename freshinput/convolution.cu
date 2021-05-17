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
#define FWIDTH 5
#define FHEIGHT 5
#define CNUMBER 16
#define centerX (FWIDTH/2)
#define centerY (FHEIGHT/2)

__shared__ float input[16][16][16];//[CHEIGHT][CWIDTH][CNUMBER]

__global__ void convolute_kernel(float *IC, float*F, float *OC) {


    //int threadIdx.x = threadthreadIdx.x.x;
    //int threadIdx.y = threadthreadIdx.x.y;
    //int centerY = FHEIGHT / 2;
    //int centerX = FWIDTH / 2;
    register float filter[FHEIGHT][FWIDTH];
    register float output[(CHEIGHT / BLOCK_SIZE) * CWIDTH] = {0};

    for(int i = 0; i < CHEIGHT; i++)
        input[i][threadIdx.y][threadIdx.x] = IC[i * CNUMBER * CWIDTH + threadIdx.y * CNUMBER + threadIdx.x];
    __syncthreads();

    for(int i = 0; i < FHEIGHT; i++)
    {
        for(int j = 0; j < FWIDTH; j++)
        {
            filter[i][j] = F[i * CNUMBER * FWIDTH + j * CNUMBER + threadIdx.x];
        }
    }
    // each threadIdx.y


    if(threadIdx.y >= centerY && threadIdx.y < CHEIGHT - centerY)
    {
        int f = 0;
        
        for(int j = threadIdx.y - 2; j <= threadIdx.y + 2; j++)
        {
            
            for(int i = FWIDTH - 1; i <= CWIDTH - FWIDTH; i++)
            {
                float in = input[j][i][threadIdx.x];
                int t = 0;
                for(int n = i - 2; n <= i + 2; n++)
                {
                    output[n] += in * filter[f][t++];
                }

            }
            int i = FWIDTH - 2;
            float in = input[j][i][threadIdx.x];
            int t = 1;
            #pragma unroll
            for(int n = i - 1; n <= i + 2; n++)
            {
                output[n] += in * filter[f][t++];
            }
            i = FWIDTH - 3;
            in = input[j][i][threadIdx.x];
            t = 2;
            #pragma unroll
            for(int n = i; n <= i + 2; n++)
            {
                output[n] += in * filter[f][t++];
            }
            i = CWIDTH - FWIDTH + 1;
            in = input[j][i][threadIdx.x];
            t = 0;
            #pragma unroll
            for(int n = i - 2; n <= i + 1; n++)
            {
                output[n] += in * filter[f][t++];
            }
            i = CWIDTH - FWIDTH + 2;
            in = input[j][i][threadIdx.x];
            t = 0;
            #pragma unroll
            for(int n = i - 2; n <= i; n++)
            {
                output[n] += in * filter[f][t++];
            }
            f++;
        } 
        
        int f_start_y = threadIdx.y - centerY;
        for(int i = centerX; i <= FWIDTH - centerX; i++)
        {
            int f_start_x = i - centerX;
            for(int m = 0; m < FHEIGHT; m++)
            {
                for(int n = 0; n < centerX - f_start_x; n++)
                {
                    output[i] += input[f_start_y + m][f_start_x + n][threadIdx.x] * filter[m][n];
                }
            }                            
        }

        // FWIDTH edges for column
        int i = CWIDTH - FWIDTH + 1;
        int f_start_x = i - centerX;
        f_start_y = threadIdx.y - centerY;
        for(int m = 0; m < FHEIGHT; m++)
        {
            int n = 4;
            output[i] += input[f_start_y + m][f_start_x + n][threadIdx.x] * filter[m][n];
            
        } 
        i = CWIDTH - FWIDTH + 2; 
        f_start_y = threadIdx.y - centerY;
        f_start_x = i - centerX;
        for(int m = 0; m < FHEIGHT; m++)
        {
            for(int n = 3; n <= 4; n++)
            {
                output[i] += input[f_start_y + m][f_start_x + n][threadIdx.x] * filter[m][n];
            }
        }                            




        
        
        f_start_y = threadIdx.y - centerY;
        /*
        for(int i = centerX; i < CWIDTH - centerX; i++)
        {
            int f_start_x = i - centerX;
            for(int m = 0; m < FHEIGHT; m++)
            {
                for(int n = 0; n < FWIDTH; n++)
                {
                    output[i] += input[f_start_y + m][f_start_x + n][threadIdx.x] * filter[m][n];
                }
            }
        }
        */
        for(int i = 0; i < centerX; i++)
        {
            //int f_start_y = threadIdx.y - centerY;
            //int f_start_x = 0;
            
            for(int m = 0; m < FHEIGHT; m++)
            {
                int f_start_x = 0;
                for(int n = centerX - i; n < FWIDTH; n++)
                {
                    output[i] += input[f_start_y + m][f_start_x++][threadIdx.x] * filter[m][n];
                }
            }
        }
        //int l = centerX;
        for(int i = 0; i < centerX; i++)
        {
            //int f_start_y = threadIdx.y - centerY;
            int f_start_x = (CWIDTH - 1) - i - centerX;
            for(int m = 0; m < FHEIGHT; m++)
            {
                for(int n = 0; n < FWIDTH - centerX + i; n++)
                {
                    output[(CWIDTH - 1) - i] += input[f_start_y + m][f_start_x + n][threadIdx.x] * filter[m][n];
                }
            }
            //l--;
        }
    }

    if(threadIdx.y < centerY)
    {
        int f = centerY - threadIdx.y;
        
        for(int j = 0; j <= threadIdx.y + 2; j++)
        {
            
            for(int i = FWIDTH - 1; i <= CWIDTH - FWIDTH; i++)
            {
                float in = input[j][i][threadIdx.x];
                int t = 0;
                for(int n = i - 2; n <= i + 2; n++)
                {
                    output[n] += in * filter[f][t++];
                }

            }
            int i = FWIDTH - 2;
            float in = input[j][i][threadIdx.x];
            int t = 1;
            #pragma unroll
            for(int n = i - 1; n <= i + 2; n++)
            {
                output[n] += in * filter[f][t++];
            }
            i = FWIDTH - 3;
            in = input[j][i][threadIdx.x];
            t = 2;
            #pragma unroll
            for(int n = i; n <= i + 2; n++)
            {
                output[n] += in * filter[f][t++];
            }
            i = CWIDTH - FWIDTH + 1;
            in = input[j][i][threadIdx.x];
            t = 0;
            #pragma unroll
            for(int n = i - 2; n <= i + 1; n++)
            {
                output[n] += in * filter[f][t++];
            }
            i = CWIDTH - FWIDTH + 2;
            in = input[j][i][threadIdx.x];
            t = 0;
            #pragma unroll
            for(int n = i - 2; n <= i; n++)
            {
                output[n] += in * filter[f][t++];
            }
            f++;
        }         
        
        int f_start_y = threadIdx.y - centerY;
        for(int i = centerX; i <= FWIDTH - centerX; i++)
        {
            int f_start_x = i - centerX;
            for(int m = centerY - threadIdx.y; m < FHEIGHT; m++)
            {
                for(int n = 0; n < centerX - f_start_x; n++)
                {
                    output[i] += input[f_start_y + m][f_start_x + n][threadIdx.x] * filter[m][n];
                }
            }                            
        }

        // FWIDTH edges for column
        int i = CWIDTH - FWIDTH + 1;
        int f_start_x = i - centerX;
        f_start_y = threadIdx.y - centerY;
        for(int m = centerY - threadIdx.y; m < FHEIGHT; m++)
        {
            int n = 4;
            output[i] += input[f_start_y + m][f_start_x + n][threadIdx.x] * filter[m][n];
            
        } 
        i = CWIDTH - FWIDTH + 2; 
        f_start_y = threadIdx.y - centerY;
        f_start_x = i - centerX;
        for(int m = centerY - threadIdx.y; m < FHEIGHT; m++)
        {
            for(int n = 3; n <= 4; n++)
            {
                output[i] += input[f_start_y + m][f_start_x + n][threadIdx.x] * filter[m][n];
            }
        } 
        


        /*
        for(int i = centerX; i < CWIDTH - centerX; i++)
        {
            //int l = 0;
            int f_start_y = 0;
            int f_start_x = i - centerX;
            for(int m = centerY - threadIdx.y; m < FHEIGHT; m++)
            {
                for(int n = 0; n < FWIDTH; n++)
                {
                    output[i] += input[f_start_y][f_start_x + n][threadIdx.x] * filter[m][n];
                }
                f_start_y++;
            }

        }
        */
        for(int i = 0; i < centerX; i++)
        {
            int f_start_y = 0;
            //int f_start_x = i;
            for(int m = centerY - threadIdx.y; m < FHEIGHT; m++)
            {
                int f_start_x = 0;
                for(int n = centerX - i; n < FWIDTH; n++)
                {
                    output[i] += input[f_start_y][f_start_x++][threadIdx.x] * filter[m][n];
                }
                f_start_y++;
            }
        }
        for(int i = 0; i < centerX; i++)
        {
            int f_start_y = 0;
            int f_start_x = (CWIDTH - 1) - i - centerX;
            for(int m = centerY - threadIdx.y; m < FHEIGHT; m++)
            {
                for(int n = 0; n < FWIDTH - centerX + i; n++)
                {
                    output[(CWIDTH - 1) - i] += input[f_start_y][f_start_x + n][threadIdx.x] * filter[m][n];
                }
                f_start_y++;
            }
        }
    }



    if(threadIdx.y >= CHEIGHT - centerY)
    {
    
        int f = 0;
        
        for(int j = threadIdx.y - centerY; j <= 15; j++)
        {
            
            for(int i = FWIDTH - 1; i <= CWIDTH - FWIDTH; i++)
            {
                float in = input[j][i][threadIdx.x];
                int t = 0;
                for(int n = i - 2; n <= i + 2; n++)
                {
                    output[n] += in * filter[f][t++];
                }

            }
            int i = FWIDTH - 2;
            float in = input[j][i][threadIdx.x];
            int t = 1;
            #pragma unroll
            for(int n = i - 1; n <= i + 2; n++)
            {
                output[n] += in * filter[f][t++];
            }
            i = FWIDTH - 3;
            in = input[j][i][threadIdx.x];
            t = 2;
            #pragma unroll
            for(int n = i; n <= i + 2; n++)
            {
                output[n] += in * filter[f][t++];
            }
            i = CWIDTH - FWIDTH + 1;
            in = input[j][i][threadIdx.x];
            t = 0;
            #pragma unroll
            for(int n = i - 2; n <= i + 1; n++)
            {
                output[n] += in * filter[f][t++];
            }
            i = CWIDTH - FWIDTH + 2;
            in = input[j][i][threadIdx.x];
            t = 0;
            #pragma unroll
            for(int n = i - 2; n <= i; n++)
            {
                output[n] += in * filter[f][t++];
            }
            f++;
        }         
        // FWIDTH edges for column
        int f_start_y = threadIdx.y - centerY;
        for(int i = centerX; i <= FWIDTH - centerX; i++)
        {
            int f_start_x = i - centerX;
            for(int m = 0; m < (FHEIGHT - centerY) + CHEIGHT - threadIdx.y - 1; m++)
            {
                for(int n = 0; n < centerX - f_start_x; n++)
                {
                    output[i] += input[f_start_y + m][f_start_x + n][threadIdx.x] * filter[m][n];
                }
            }                            
        }
        
        
        
        int i = CWIDTH - FWIDTH + 1;
        int f_start_x = i - centerX;
        f_start_y = threadIdx.y - centerY;
        for(int m = 0; m < (FHEIGHT - centerY) + CHEIGHT - threadIdx.y - 1; m++)
        {
            int n = 4;
            output[i] += input[f_start_y + m][f_start_x + n][threadIdx.x] * filter[m][n];
            
        } 
        i = CWIDTH - FWIDTH + 2; 
        f_start_y = threadIdx.y - centerY;
        f_start_x = i - centerX;
        for(int m = 0; m < (FHEIGHT - centerY) + CHEIGHT - threadIdx.y - 1; m++)
        {
            for(int n = 3; n <= 4; n++)
            {
                output[i] += input[f_start_y + m][f_start_x + n][threadIdx.x] * filter[m][n];
            }
        }  
        


        /*
        for(int i = centerX; i < CWIDTH - centerX; i++)
        {
            int f_start_y = threadIdx.y - centerY;
            int f_start_x = i - centerX;
            for(int m = 0; m < (FHEIGHT - centerY) + CHEIGHT - threadIdx.y - 1; m++)
            {
                for(int n = 0; n < FWIDTH; n++)
                {
                    output[i] += input[f_start_y + m][f_start_x + n][threadIdx.x] * filter[m][n];
                }
            }
        }
        */
        for(int i = 0; i < centerX; i++)
        {
            int f_start_y = threadIdx.y - centerY;
            //int f_start_y = threadIdx.y;
            
            for(int m = 0; m < (FHEIGHT - centerY) + CHEIGHT - threadIdx.y - 1; m++)
            {
                int f_start_x = 0;
                for(int n = centerX - i; n < FWIDTH; n++)
                {
                    output[i] += input[f_start_y + m][f_start_x++][threadIdx.x] * filter[m][n];
                }
            }
        }
        for(int i = 0; i < centerX; i++)
        {
            int f_start_y = threadIdx.y - centerY;
            //int f_start_y = threadIdx.y;
            int f_start_x = (CWIDTH - 1) - i - centerX;
            for(int m = 0; m < (FHEIGHT - centerY) + CHEIGHT - threadIdx.y - 1; m++)
            {
                for(int n = 0; n < FWIDTH - centerX + i; n++)
                {
                    output[(CWIDTH - 1) - i] += input[f_start_y + m][f_start_x + n][threadIdx.x] * filter[m][n];
                }
            }
        }
    }


    /*
    if(threadIdx.x == 0 && threadIdx.y == 14)
    {
        for(int i = 0; i < CWIDTH; i++)
            printf("%.1f ",output[i]);
        printf("\n");
    }
    */
    for(int i = 0; i < CWIDTH; i++)
    {
        OC[threadIdx.y * CNUMBER * CWIDTH + i * CNUMBER + threadIdx.x] = output[i];
    }


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
    



    bool equal = true;

    double fops = 0.0;
    for (int i = 0;i < CHEIGHT; i++){
        for (int j = 0; j < CWIDTH; j++) {
            for(int k = 0; k < CNUMBER; k++) {
                fops += OC[i * CWIDTH * CNUMBER + j * CNUMBER + k];
                if(OC_cpu[k][i][j] != OC[i * CWIDTH * CNUMBER + j * CNUMBER + k])
                {
                    equal = false;
                    //printf("%0.1f ",OC[i * CWIDTH * CNUMBER + j * CNUMBER + k]);
                    //printf("%0.1f ",OC[k * CWIDTH * CNUMBER + i * CNUMBER + j]);
                    //printf("%d %d %d  %.1f != %.1f\n", k, i, j, OC_cpu[k][i][j], OC[i * CWIDTH * CNUMBER + j * CNUMBER + k]);
                        //break;
                }
                //printf("%d",OC[k * CNUMBER * CWIDTH + i * CWIDTH + j]);      
            }
            //printf("\n");
        }
        //printf("\n");
    }
    /*
    for (int i = 0;i < CNUMBER; i++){
        for (int j = 0; j < CHEIGHT; j++) {
            for(int k = 0; k < CWIDTH; k++) {
                printf("%d ",OC[j * CNUMBER * CWIDTH + k * CNUMBER + i]);
            }
            printf("\n");
        }
        printf("\n");
    }
    */
    printf("Bandwidth = %f GFLOPS/s\n", (2 * fops) / (m * 1000000));

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