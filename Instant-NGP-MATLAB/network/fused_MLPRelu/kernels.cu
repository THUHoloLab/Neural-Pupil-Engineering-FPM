#include "addon.h"
#include <cuda_runtime.h>

__global__ void fusedGemmBiasReluTiled(
    const float* __restrict__ A,  // [M x K], col-major
    const float* __restrict__ X,  // [K x N], col-major
    const float* __restrict__ B,  // [N], column bias
    float* Y,                     // [M x N], col-major
    int M, int K, int N
) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Xsub[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // M direction
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // N direction

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A: A[k * M + row]
        if ((t * TILE_SIZE + threadIdx.x) < K && row < M)
            Asub[threadIdx.y][threadIdx.x] = A[(t * TILE_SIZE + threadIdx.x) * M + row];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of X: X[col * K + k]
        if ((t * TILE_SIZE + threadIdx.y) < K && col < N)
            Xsub[threadIdx.y][threadIdx.x] = X[col * K + (t * TILE_SIZE + threadIdx.y)];
        else
            Xsub[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += Asub[threadIdx.y][k] * Xsub[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        acc += B[col];  // Column-wise bias
        Y[col * M + row] = fmaxf(acc, 0.0f);  // ReLU
    }
}