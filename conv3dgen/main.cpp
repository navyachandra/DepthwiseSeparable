#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <iostream>
#define CWIDTH 32
#define CHEIGHT 32
#define FWIDTH 3
#define FHEIGHT 3
#define CNUMBER 32


using namespace std;
void cudaconvolute(float* ic, float* f, float* oc, float*** cpuoc);
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

int main() {


    float* IC = new float[CNUMBER * CHEIGHT * CWIDTH];
    float* F = new float[CNUMBER * FHEIGHT * FWIDTH];
    float* OC = new float[CNUMBER * CHEIGHT * CWIDTH];

    // filter
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
                kernel[k][i][j] = j;
                *(F + k * FHEIGHT * FWIDTH + i * FWIDTH + j) = kernel[k][i][j];
            }
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
                *(IC + k * CWIDTH * CHEIGHT + i * CWIDTH + j) = matrixIn[k][i][j];

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

    convolute(output, matrixIn, kernel);
    printCudaInfo();
    cudaconvolute(IC, F, OC, output);
    for(int k = 0; k < CNUMBER; k++)
    {  
        for(int i = 0; i < CHEIGHT; i++)
        {
            for(int j = 0; j < CWIDTH; j++)
            {
                printf("%0.1f ",output[k][i][j]);
                //if(*(IC + k * CWIDTH * CHEIGHT + i * CWIDTH + j) != matrixIn[k][i][j])
                //    printf("not equal");
            }
            printf("\n");
        }
        printf("\n");
    }
    delete [] IC;
    delete [] F;
    delete [] OC;
    delete [] kernel;
    delete [] matrixIn;
    delete [] output;



    return 0;

}
