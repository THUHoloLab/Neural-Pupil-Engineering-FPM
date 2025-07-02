#pragma once

#include "addon/helpers.cuh"

__global__ void hashEncoding_Fwd_kernel(
    const float2 * __restrict__ xys,
    const float * __restrict__ embedding,
    const float * __restrict__ bbox,
    const uint32_t * __restrict__ hash_offsets,
    const uint32_t * __restrict__ hash_map_sizes,
    const dim3 output_sz,
    const float log_scale, // (log(res_high) - log(res_low)) / (level - 1)
    const float base_res,
    const uint32_t feature_dim,
    // outputs
    float * __restrict__ output_embedding
);

__global__ void hashEncoding_Bwd_kernel(
    const float2 * __restrict__ xys,
    const float * __restrict__ dl_doutput,
    const float * __restrict__ bbox,
    const uint32_t * __restrict__ hash_offsets,
    const uint32_t * __restrict__ hash_map_sizes,
    const dim3 output_sz,
    const float log_scale, // (log(res_high) - log(res_low)) / (level - 1)
    const float base_res,
    const uint32_t feature_dim,
    float * __restrict__ dl_dembedding
);
