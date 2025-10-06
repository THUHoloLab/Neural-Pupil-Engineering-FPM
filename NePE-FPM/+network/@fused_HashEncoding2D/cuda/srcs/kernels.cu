#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

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
){
    uint32_t batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t level_id = blockIdx.y; // page id

    if (batch_id >= output_sz.y) {
        return;
    }

    float scale = grid_resolution((int) level_id, log_scale, base_res);
    uint32_t resolution = uint32_t(ceilf(scale)) + 1;
    float pos[2] = {xys[batch_id].x, xys[batch_id].y};
    uint32_t pos_grid[2];
    #pragma unroll
    for(uint32_t idx = 0; idx < 2; ++ idx){
        float bbox_min = bbox[idx * 2 + 0];
        float normalized_pos = (pos[idx] - bbox_min) / 
                               (bbox[idx * 2 + 1] - bbox_min) * scale;
        // normalized_pos = max(min(normalized_pos,1.0f),0.0f);
        float temp_pos = __floorf(normalized_pos);
        pos_grid[idx] = (uint32_t) temp_pos;
        pos[idx] = normalized_pos - temp_pos;
    }

    uint32_t map_size = hash_map_sizes[level_id];
    uint32_t offsets  = hash_offsets[level_id];

    // float feature1 = 0.0f;
    // float feature2 = 0.0f;
    float feature[dim_max] = {0.f,0.f,0.f,0.f};
    
    #pragma unroll
    for(uint32_t idx = 0; idx < 4; ++idx){
        float w = 1.0f;
        uint32_t loc_pos[2];
        #pragma unroll
        for(uint32_t dim = 0; dim < 2; ++dim){
            if ((idx & (1 << dim)) == 0){
                loc_pos[dim] = pos_grid[dim];
                w *= 1 - pos[dim];
            }else{
                loc_pos[dim] = pos_grid[dim] + 1;
                w *= pos[dim];
            }
        }
        uint32_t hash_idx = grid_index(
            map_size,
            resolution,
            loc_pos[0],
            loc_pos[1]
        );

        uint32_t table_index = (offsets + hash_idx) * feature_dim;
        for (uint32_t fdims = 0; fdims < feature_dim; fdims ++){
            feature[fdims] = fma(w, embedding[table_index + fdims], feature[fdims]);
        }
        // feature1 = fma(w, embedding[table_index + 0], feature1);
        // feature2 = fma(w, embedding[table_index + 1], feature2);
    }
    uint32_t batch_offset = batch_id * output_sz.x;
    for (uint32_t fdims = 0; fdims < feature_dim; fdims ++){
        output_embedding[batch_offset + level_id * feature_dim + fdims] = feature[fdims];
    }
}

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
){
    uint32_t batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t level_id = blockIdx.y; // page id

    if (batch_id >= output_sz.y) {
        return;
    }

    float scale = grid_resolution((int) level_id, log_scale, base_res);
    uint32_t resolution = (uint32_t) ceilf(scale) + 1;
    float pos[2] = {xys[batch_id].x, xys[batch_id].y};

    uint32_t pos_grid[2];

    #pragma unroll
    for(uint32_t idx = 0; idx < 2; ++ idx){
        float bbox_min = bbox[idx * 2 + 0];
        float normalized_pos = (pos[idx] - bbox_min) / 
                               ((bbox[idx * 2 + 1] - bbox_min) / scale);
        // normalized_pos = max(min(normalized_pos,1.0f),0.0f);
        float temp_pos = floorf(normalized_pos);
        pos_grid[idx] = (uint32_t) temp_pos;
        pos[idx] = normalized_pos - temp_pos;
    }

    uint32_t map_size = hash_map_sizes[level_id];
    uint32_t offsets  = hash_offsets[level_id];

    uint32_t batch_offset = batch_id * output_sz.x + level_id * feature_dim;
    float dl_output[8];
    #pragma unroll
    for (uint32_t fdims = 0; fdims < feature_dim; fdims ++){
        dl_output[fdims] = dl_doutput[batch_offset + fdims];
    }
    // float dl_output1 = dl_doutput[batch_offset + 0];
    // float dl_output2 = dl_doutput[batch_offset + 1];
    #pragma unroll
    for(uint32_t idx = 0; idx < 4; ++idx){
        float w = 1.0f;
        uint32_t loc_pos[2];
        #pragma unroll
        for(uint32_t dim = 0; dim < 2; ++dim){
            if ((idx & (1 << dim)) == 0){
                loc_pos[dim] = pos_grid[dim];
                w *= 1 - pos[dim];
            }else{
                loc_pos[dim] = pos_grid[dim] + 1;
                w *= pos[dim];
            }
        }
 
        uint32_t hash_idx = grid_index(
            map_size,
            resolution,
            loc_pos[0],
            loc_pos[1]
        );

        uint32_t base_id = (offsets + hash_idx) * feature_dim;
        float *temp = (float *) dl_dembedding;

        #pragma unroll
        for (uint32_t fdims = 0; fdims < feature_dim; fdims ++){
            atomicAdd(temp + base_id + fdims, w * dl_output[fdims]);
        }
        // atomicAdd(temp + base_id + 0, w * dl_output1);
        // atomicAdd(temp + base_id + 1, w * dl_output2);
    }
}
