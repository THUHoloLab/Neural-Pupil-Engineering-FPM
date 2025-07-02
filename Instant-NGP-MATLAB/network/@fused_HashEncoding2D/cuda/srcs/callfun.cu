#include "addon/helpers.cuh"
#include "callfun.cuh"

__host__ dim3 size2dim3( const mxGPUArray * in){
    const mwSize *sz = mxGPUGetDimensions(in);
    const int  dim = (int) mxGPUGetNumberOfDimensions(in);
    dim3 imgSz;
    imgSz = {(unsigned) sz[0], (unsigned) sz[1], 1};
    if (dim > 2){
        imgSz.z = (unsigned) sz[2];
    }
    return imgSz;
}

void Call_Fwd(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const * prhs[]
){
    const mxGPUArray * xys_batch;
    const mxGPUArray * embedding;
    const mxGPUArray * bound_box;
    const mxGPUArray * hash_mapsizes;
    const mxGPUArray * hash_offsets;

    mxGPUArray * output_embedding;

    mxInitGPU();

    CHECK_THROW(mxIsGPUArray(prhs[0]));
    CHECK_THROW(mxIsGPUArray(prhs[1]));
    CHECK_THROW(mxIsGPUArray(prhs[2]));
    CHECK_THROW(mxIsGPUArray(prhs[3]));
    CHECK_THROW(mxIsGPUArray(prhs[4]));

    xys_batch = mxGPUCreateFromMxArray(prhs[0]);
    embedding = mxGPUCreateFromMxArray(prhs[1]);
    bound_box = mxGPUCreateFromMxArray(prhs[2]);
    hash_mapsizes = mxGPUCreateFromMxArray(prhs[3]);
    hash_offsets = mxGPUCreateFromMxArray(prhs[4]);

    const unsigned total_levels = (const unsigned) mxGetPr(prhs[5])[0];
    const float log_scale = (const float) mxGetPr(prhs[6])[0];
    const float base_res  = (const float) mxGetPr(prhs[7])[0];
    const uint32_t feature_dim = (const uint32_t) mxGetPr(prhs[8])[0];

    const dim3 xys_sz = size2dim3(xys_batch);
    const dim3 embedding_sz = size2dim3(embedding);

    const float2 *d_xys_batch = (const float2 * __restrict__ ) mxGPUGetDataReadOnly(xys_batch);
    const float *d_embedding = (const float * __restrict__ ) mxGPUGetDataReadOnly(embedding);
    const float *d_bound_box = (const float * __restrict__ ) mxGPUGetDataReadOnly(bound_box);
    const uint32_t *d_hash_mapsizes = (const uint32_t * __restrict__ ) mxGPUGetDataReadOnly(hash_mapsizes);
    const uint32_t *d_hash_offsets = (const uint32_t * __restrict__ ) mxGPUGetDataReadOnly(hash_offsets);

    CHECK_THROW((size_t) xys_sz.x == 2); // 2D positions
    // mexPrintf("Forward, xys_batch size is %u %u %u \n", xys_sz.x, xys_sz.y, xys_sz.z);
    // mexPrintf("embedding size is %u %u %u \n", embedding_sz.x, embedding_sz.y, embedding_sz.z);
    dim3 N_BLOCKS = {
        (unsigned) (xys_sz.y + N_THREADS - 1) / N_THREADS,
        total_levels
    };

    const mwSize sz[2] = {feature_dim * total_levels, xys_sz.y};
    dim3 output_sz = {(unsigned) sz[0], (unsigned) sz[1], 1};
    output_embedding = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(xys_batch),
        sz,
        mxSINGLE_CLASS,
        mxREAL,
        MX_GPU_INITIALIZE_VALUES
    );  

    float * d_output_embedding = (float *__restrict__) mxGPUGetData(output_embedding);
    hashEncoding_Fwd_kernel<<<N_BLOCKS, N_THREADS>>>(
        d_xys_batch,        // const float2 * __restrict__ xys,
        d_embedding,        // const float * __restrict__ embedding,
        d_bound_box,        // const float * __restrict__ bbox,
        d_hash_offsets,     // const uint32_t * __restrict__ hash_offsets,
        d_hash_mapsizes,    // const uint32_t * __restrict__ hash_map_sizes,
        output_sz,          // const dim3 output_sz,
        log_scale,          // const float log_scale, // (log(res_high) - log(res_low)) / (level - 1)
        base_res,           // const float base_res,
        feature_dim,        // const uint32_t feature_dim,
        // outpus
        d_output_embedding// float * __restrict__ output_embedding
    );
    // cudaDeviceSynchronize();
    plhs[0] = mxGPUCreateMxArrayOnGPU(output_embedding);

    mxGPUDestroyGPUArray(xys_batch);
    mxGPUDestroyGPUArray(embedding);
    mxGPUDestroyGPUArray(bound_box);
    mxGPUDestroyGPUArray(hash_mapsizes);
    mxGPUDestroyGPUArray(hash_offsets);
    mxGPUDestroyGPUArray(output_embedding);
}

void Call_Bwd(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const * prhs[]
){
    const mxGPUArray * xys_batch;
    const mxGPUArray * embedding;
    const mxGPUArray * dl_doutput;
    const mxGPUArray * bound_box;
    const mxGPUArray * hash_mapsizes;
    const mxGPUArray * hash_offsets;

    mxGPUArray * grad_embedding;
    mxInitGPU();

    CHECK_THROW(mxIsGPUArray(prhs[0]));
    CHECK_THROW(mxIsGPUArray(prhs[1]));
    CHECK_THROW(mxIsGPUArray(prhs[2]));
    CHECK_THROW(mxIsGPUArray(prhs[3]));
    CHECK_THROW(mxIsGPUArray(prhs[4]));
    CHECK_THROW(mxIsGPUArray(prhs[5]));

    xys_batch   = mxGPUCreateFromMxArray(prhs[0]);
    embedding   = mxGPUCreateFromMxArray(prhs[1]);
    dl_doutput  = mxGPUCreateFromMxArray(prhs[2]);
    bound_box   = mxGPUCreateFromMxArray(prhs[3]);
    hash_mapsizes = mxGPUCreateFromMxArray(prhs[4]);
    hash_offsets = mxGPUCreateFromMxArray(prhs[5]);


    unsigned total_levels = (unsigned) mxGetPr(prhs[6])[0];
    float log_scale = (float) mxGetPr(prhs[7])[0];
    float base_res  = (float) mxGetPr(prhs[8])[0];
    uint32_t feature_dim = (uint32_t) mxGetPr(prhs[9])[0];

    dim3 xys_sz = size2dim3(xys_batch);
    dim3 embedding_sz = size2dim3(embedding);
    dim3 dl_doutput_sz = size2dim3(dl_doutput);
    // mexPrintf("Backward, xys_batch size is %u %u %u \n", xys_sz.x, xys_sz.y, xys_sz.z);
    // mexPrintf("dl_doutput_sz size is %u %u %u \n", dl_doutput_sz.x, dl_doutput_sz.y, dl_doutput_sz.z);
    CHECK_THROW((size_t) xys_sz.x == 2);
    CHECK_THROW((size_t) xys_sz.y == (size_t) dl_doutput_sz.y);
    CHECK_THROW((size_t) dl_doutput_sz.x == (size_t) (feature_dim * total_levels));

    const float2 *d_xys_batch = (const float2 * __restrict__ ) mxGPUGetDataReadOnly(xys_batch);
    const float *d_dl_doutput = (const float * __restrict__ ) mxGPUGetDataReadOnly(dl_doutput);
    const float *d_bound_box = (const float * __restrict__ ) mxGPUGetDataReadOnly(bound_box);
    const uint32_t *d_hash_mapsizes = (const uint32_t * __restrict__ ) mxGPUGetDataReadOnly(hash_mapsizes);
    const uint32_t *d_hash_offsets = (const uint32_t * __restrict__ ) mxGPUGetDataReadOnly(hash_offsets);

    int batch_size = (int) xys_sz.y;
    dim3 N_BLOCKS = {
        (unsigned) (batch_size + N_THREADS - 1) / N_THREADS,
        total_levels
    };
    // gradient of embedding
    grad_embedding = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(embedding),
        mxGPUGetDimensions(embedding),
        mxSINGLE_CLASS,
        mxREAL,
        MX_GPU_INITIALIZE_VALUES
    );  

    float * d_grad_embedding = (float *__restrict__) mxGPUGetData(grad_embedding);

    hashEncoding_Bwd_kernel<<<N_BLOCKS, N_THREADS>>>(
        d_xys_batch,        // const float2 * __restrict__ xys,
        d_dl_doutput,       // const float * __restrict__ dl_doutput,
        d_bound_box,        // const float * __restrict__ bbox,
        d_hash_offsets,     // const uint32_t * __restrict__ hash_offsets,
        d_hash_mapsizes,    // const uint32_t * __restrict__ hash_map_sizes,
        dl_doutput_sz,      // const dim3 output_sz,
        log_scale,          // const float log_scale, // (log(res_high) - log(res_low)) / (level - 1)
        base_res,           // const float base_res,
        feature_dim,        // const uint32_t feature_dim,
        // output
        d_grad_embedding    // float * __restrict__ dl_dembedding
    );
    // cudaDeviceSynchronize();
    plhs[0] = mxGPUCreateMxArrayOnGPU(grad_embedding);
    mxGPUDestroyGPUArray(xys_batch); 
    mxGPUDestroyGPUArray(embedding);
    mxGPUDestroyGPUArray(bound_box); 
    mxGPUDestroyGPUArray(dl_doutput); 
    mxGPUDestroyGPUArray(hash_mapsizes); 
    mxGPUDestroyGPUArray(hash_offsets); 
    mxGPUDestroyGPUArray(grad_embedding); 
}