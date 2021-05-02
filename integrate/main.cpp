#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <iostream>
#define CWIDTH 16
#define CHEIGHT 16
#define FWIDTH 3
#define FHEIGHT 3
#define CNUMBER 16
#define COUT 16

using namespace std;
void cudaconvolute(float* ic, float* f, float* fp, float* soc, float*** cpuoc, float *** cpusoc);
void printCudaInfo();
void convolute(float *** output, float *** input, float *** kernel)
{

    int kCenterX = FWIDTH / 2;
    int kCenterY = FHEIGHT / 2;

    for(int k = 0; k < CNUMBER; k++)
    {
        for (int i = 0; i < CHEIGHT; ++i)              // rows
        {
            for (int j = 0; j < CWIDTH; ++j)          // columns
            {
                for (int m = 0; m < FHEIGHT; ++m)     // kernel rows
                {

                    for (int n = 0; n < FWIDTH; ++n) // kernel columns
                    {

                        // index of input, used for checking boundary
                        int ii = i + (m - kCenterY);
                        int jj = j + (n - kCenterX);

                        // ignore input samples which are out of bound
                        if (ii >= 0 && ii < CHEIGHT && jj >= 0 && jj < CWIDTH)
                            output[k][i][j] += input[k][ii][jj] * kernel[k][m][n];
                    }
                }
            }
        }
    }
}
void pointwise(float *** sepOut, float *** output, float ** filter)
{

    for(int l = 0; l < COUT; l++)
    {
        for (int i = 0; i < CHEIGHT; ++i)              // rows
        {
            for (int j = 0; j < CWIDTH; ++j)          // columns
            {
                for (int k = 0; k < CNUMBER; ++k)     // kernel rows
                {
                    sepOut[l][i][j] += output[k][i][j] * filter[l][k];
                }
            }
        }
    }    
}

int main() 
{


    float* IC = new float[CNUMBER * CHEIGHT * CWIDTH];
    float* F = new float[CNUMBER * FHEIGHT * FWIDTH];
    float* FP = new float[CNUMBER * COUT];
    float* OC = new float[CNUMBER * CHEIGHT * CWIDTH];
    float* SOC = new float[COUT * CHEIGHT * CWIDTH];

    // depthwise filter
    float *** kernel = new float**[CNUMBER];
    for(int i = 0; i < CNUMBER; ++i)
    {
        kernel[i] = new float*[FHEIGHT];
    }
    for(int k = 0; k < CNUMBER; k++)
    {
        for(int i = 0; i < FHEIGHT; i++)
        {
            kernel[k][i] = new float[FWIDTH];
        }
    }    
    for(int k = 0; k < CNUMBER; k++)
    {
        for(int i = 0; i < FHEIGHT; i++)
        {
            for(int j = 0; j < FWIDTH; j++)
            {
                kernel[k][i][j] = 1;
                //*(F + k * FHEIGHT * FWIDTH + i * FWIDTH + j) = kernel[k][i][j];
            }
        }
    }
    for (int i = 0; i < FHEIGHT; i++)
    {
        for (int j = 0; j < FWIDTH; j++)
        {
            for (int k = 0; k < CNUMBER; k++)
            {
                *(F + i * FWIDTH * CNUMBER + j * CNUMBER + k) = kernel[k][i][j];
            }
        }
    }    
    
    // pointwise filter
    float ** filter = new float*[COUT];
    for(int i = 0; i < COUT; ++i)
    {
        filter[i] = new float[CNUMBER];
    }
    for(int i = 0; i < COUT; i++)
    {
        for(int j = 0; j < CNUMBER; j++)
        {
            filter[i][j] = 1;
            *(FP + i * CNUMBER + j) = filter[i][j];
        }
    }

    // input matrices
    float *** matrixIn = new float**[CNUMBER];
    for(int i = 0; i < CNUMBER; ++i)
    {
        matrixIn[i] = new float*[CHEIGHT];
    }
    for(int k = 0; k < CNUMBER; k++)
    {
        for(int i = 0; i < CHEIGHT; i++)
        {
            matrixIn[k][i] = new float[CWIDTH];
        }
    } 
    for(int k = 0; k < CNUMBER; k++)
    {
        for(int i = 0; i < CHEIGHT; i++)
        {
            for(int j = 0; j < CWIDTH; j++)
            {    
                matrixIn[k][i][j] = 1;
                //*(IC + k * CWIDTH * CHEIGHT + i * CWIDTH + j) = matrixIn[k][i][j];

            }   
        }
    }

    for (int i = 0; i < CHEIGHT; i++)
    {
        for (int j = 0; j < CWIDTH; j++)
        {
            for (int k = 0; k < CNUMBER; k++)
            {
                *(IC + i * CWIDTH * CNUMBER + j * CNUMBER + k) = matrixIn[k][i][j];
            }
        }
    }

    // output matrices
    float *** output = new float**[CNUMBER];
    for(int i = 0; i < CNUMBER; ++i)
    {
        output[i] = new float*[CHEIGHT];
    }
    for(int k = 0; k < CNUMBER; k++)
    {
        for(int i = 0; i < CHEIGHT; i++)
        {
            output[k][i] = new float[CWIDTH];
        }
    } 
    for(int k = 0; k < CNUMBER; k++)
    {   
        for(int i = 0; i < CHEIGHT; i++)
        {    
            for(int j = 0; j < CWIDTH; j++)
            {
                output[k][i][j] = 0;
                *(OC + k * CHEIGHT * CWIDTH + i * CWIDTH + j) = 0;
            }    
        }
    }
    //pointwise output matrices
    float *** sepOut = new float**[COUT];
    for(int i = 0; i < COUT; ++i)
    {
        sepOut[i] = new float*[CHEIGHT];
    }
    for(int k = 0; k < COUT; k++)
    {
        for(int i = 0; i < CHEIGHT; i++)
        {
            sepOut[k][i] = new float[CWIDTH];
        }
    } 
    for(int k = 0; k < COUT; k++)
    {   
        for(int i = 0; i < CHEIGHT; i++)
        {    
            for(int j = 0; j < CWIDTH; j++)
            {
                sepOut[k][i][j] = 0;
                *(SOC + k * CHEIGHT * CWIDTH + i * CWIDTH + j) = 0;
            }    
        }
    }

    convolute(output, matrixIn, kernel);
    pointwise(sepOut, output, filter);
    printCudaInfo();
    cudaconvolute(IC, F, FP, SOC, output, sepOut);
    for(int k = 0; k < COUT; k++)
    {  
        for(int i = 0; i < CHEIGHT; i++)
        {
            for(int j = 0; j < CWIDTH; j++)
            {
                //printf("%0.1f ",sepOut[k][i][j]);
                //if(*(IC + k * CWIDTH * CHEIGHT + i * CWIDTH + j) != matrixIn[k][i][j])
                //    printf("not equal");
            }
            //printf("\n");
        }
        //printf("\n");
    }
    delete [] IC;
    delete [] F;
    delete [] OC;
    delete [] FP;
    delete [] SOC;
    delete [] kernel;
    delete [] matrixIn;
    delete [] output;
    delete [] filter;
    delete [] sepOut;


    return 0;

}
