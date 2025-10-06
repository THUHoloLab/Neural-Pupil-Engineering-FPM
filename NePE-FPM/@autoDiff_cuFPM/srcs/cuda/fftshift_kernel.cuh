#include <cuda_runtime.h>

__global__ void cufftShift_2D_kernel(
    creal32_t* __restrict__ data, 
    int N
){
    unsigned xIndex = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned zIndex = blockIdx.z;  

    const bool inside = (xIndex < N) && (yIndex < N);

    if(inside){
        // 2D Slice & 1D Line
        int sSlice = N * N;
        // Transformations Equations
        int sEq1 = (sSlice + N) / 2;
        int sEq2 = (sSlice - N) / 2;

        // Thread Index Converted into 1D Index
        int index = (yIndex * N) + xIndex + sSlice * zIndex;
        creal32_t regTemp;
        if (xIndex < N / 2){
            if (yIndex < N / 2){
                regTemp = data[index];
                // First Quad
                data[index] = data[index + sEq1];
                // Third Quad
                data[index + sEq1] = regTemp;
            }
        }else{
            if (yIndex < N / 2){
                regTemp = data[index];
                // Second Quad
                data[index] = data[index + sEq2];
                // Fourth Quad
                data[index + sEq2] = regTemp;
            }
        }
    }
}