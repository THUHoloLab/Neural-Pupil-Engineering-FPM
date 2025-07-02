// #include "addon.h"
#include "mex.h"
#include "mxGPUArray.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

using mxGPUArray_t = mxGPUArray;

void mexFunction(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const * prhs[]
){
    const mxGPUArray_t * A; // [M x K]
    const mxGPUArray_t * X; // [K x N]
    mxGPUArray_t * B; // [M]
    mxGPUArray_t * Y;
    const float *d_X;
    const float *d_A;
    float *d_B;
    float *d_Y;
    mxInitGPU();

    A = mxGPUCreateFromMxArray(prhs[0]);
    X = mxGPUCreateFromMxArray(prhs[1]);
    B = mxGPUCopyFromMxArray(prhs[2]);

    const mwSize *szA = mxGPUGetDimensions(A);
    const mwSize *szX = mxGPUGetDimensions(X);
    const mwSize *szB = mxGPUGetDimensions(B);

    mwSize szY[2] = {szA[0],szX[1]};

    Y = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(A),
        szY,
        mxSINGLE_CLASS,
        mxREAL,
        MX_GPU_INITIALIZE_VALUES
    ); 

    d_A = (const float * __restrict__ ) mxGPUGetDataReadOnly(A);
    d_X = (const float * __restrict__ ) mxGPUGetDataReadOnly(X);
    d_B = (float * __restrict__ ) mxGPUGetData(B);
    d_Y = (float * __restrict__ ) mxGPUGetData(Y);
    // dim3 N_THREADS(TILE_SIZE, TILE_SIZE);
    // dim3 N_BLOCKS(((unsigned) szX[1] + TILE_SIZE - 1)/TILE_SIZE, 
    //               ((unsigned) szA[0] + TILE_SIZE - 1)/TILE_SIZE);

    // fusedGemmBiasReluTiled<<<N_BLOCKS, N_THREADS>>>(
    //     d_A, d_X, d_B, d_Y, (unsigned) szA[0], (unsigned) szX[0], (unsigned) szX[1]
    // );
    unsigned M = (unsigned) szA[0];
    unsigned K = (unsigned) szA[1];
    unsigned N = (unsigned) szX[1];

    const float alpha = 1.0f, beta = 0.0f;
    // Y = alpha * A * X + beta * B
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,  // A 和 X 都不转置
                M, N, K,                   // Y 维度 M×N, A 维度 M×K, X 维度 K×N
                &alpha,
                d_A, M,                      // A 的 leading dimension (lda)
                d_X, K,                      // X 的 leading dimension (ldx)
                &beta,
                d_Y, M); 
                
    cublasDestroy(handle);

    plhs[0] = mxGPUCreateMxArrayOnGPU(Y);

    mxGPUDestroyGPUArray(Y);
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(X);
    mxGPUDestroyGPUArray(B);
}