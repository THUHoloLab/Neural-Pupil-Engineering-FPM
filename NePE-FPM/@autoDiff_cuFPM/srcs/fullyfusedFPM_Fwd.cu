#include "mex.h"
#include "cuda/mxGPUArray.h"
#include "cuda/kernels.cuh"
#include <cuda_runtime.h>

static __global__ void ifftCorrection_sub(
    creal32_t *__restrict__ input,
    dim3 imgSz
){
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z;  

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);
    if(inside){
        unsigned page_id = idz * (imgSz.x * imgSz.y);
        unsigned pix_id = idx * imgSz.y + idy;

        creal32_t temp = input[pix_id + page_id];
        float ratio = 1.0f / (float) (imgSz.x * imgSz.y);
        temp.re = temp.re * ratio;
        temp.im = temp.im * ratio;
        input[pix_id + page_id] = temp;
    }
}

dim3 size2dim3( const mxGPUArray * in){
    const mwSize *sz = mxGPUGetDimensions(in);
    const int  dim = (int) mxGPUGetNumberOfDimensions(in);
    dim3 imgSz;
    imgSz = {(unsigned) sz[1], (unsigned) sz[0], 1};
    if (dim > 2){
        imgSz.z = (unsigned) sz[2];
    }
    return imgSz;
}

void mexFunction(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const *  __restrict__ prhs[]
){
    const creal32_t * d_pupil;
    const int2 * d_ledIdx;
    mxInitGPU();

    mxGPUArray_t * wavefront1   = mxGPUCopyFromMxArray(prhs[0]); // sample wavefront
    const mxGPUArray_t * pupil  = mxGPUCreateFromMxArray(prhs[1]); // pupil wavefront
    const mxGPUArray_t * obseY  = mxGPUCreateFromMxArray(prhs[2]); // observed intensity
    const mxGPUArray_t * ledIdx = mxGPUCreateFromMxArray(prhs[3]); // LED position in pixel

    d_pupil  = getGPUDataRO<creal32_t>(pupil);
    d_ledIdx = getGPUDataRO<int2>(ledIdx);

    mxGPUArray_t * latentZ = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(obseY), 
        mxGPUGetDimensions(obseY),                   
        mxSINGLE_CLASS, 
        mxCOMPLEX, 
        MX_GPU_INITIALIZE_VALUES
    );

    mxGPUArray_t * subwave = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(obseY), 
        mxGPUGetDimensions(obseY),                   
        mxSINGLE_CLASS, 
        mxCOMPLEX, 
        MX_GPU_INITIALIZE_VALUES
    );

    creal32_t * d_wavefront1 = (creal32_t * __restrict__ ) mxGPUGetData(wavefront1);
    creal32_t * d_latentZ    = (creal32_t * __restrict__ ) mxGPUGetData(latentZ);
    creal32_t * d_subwave    = (creal32_t * __restrict__ ) mxGPUGetData(subwave);

    // get image size                            
    dim3 imLs_sz = size2dim3(obseY);
    dim3 imHs_sz = size2dim3(wavefront1);

    // FFT_sample_forward(imHs_sz.x,imHs_sz.y,d_wavefront1);

    dim3 N_THREADS = {BLOCK_X,BLOCK_Y,1};
    dim3 N_BLOCKS = {
        (unsigned) ((imHs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imHs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) 1
    };
    cufftHandle plan;
    cufftPlan2d(&plan, imHs_sz.x, imHs_sz.y, CUFFT_C2C);
    cufftExecC2C(
        plan, 
        (cufftComplex *)&d_wavefront1[0], 
        (cufftComplex *)&d_wavefront1[0],
        CUFFT_FORWARD
    );
    cufftDestroy(plan);
    cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(d_wavefront1,imHs_sz.x);

    N_BLOCKS = {
        (unsigned) ((imLs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imLs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) imLs_sz.z
    };
    getSubpupil<<<N_BLOCKS, N_THREADS>>>(
        d_wavefront1, 
        d_pupil,
        d_ledIdx,
        imLs_sz,
        imHs_sz,
        d_subwave,
        d_latentZ
    );
    cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(
        d_latentZ, (unsigned) 
        imLs_sz.x
    );
    cufftHandle plan_many;
    int inembed[2];
    for (int i{0}; i < 2; i++) {
        inembed[i] = (int) imLs_sz.x;
    }
    cufftPlanMany(
        &plan_many, 
        2, 
        &inembed[0], // n
        &inembed[0], // inembed
        1,           // istride
        (int) imLs_sz.x * (int) imLs_sz.y, // idist
        &inembed[0], // inembed
        1,           // istride
        (int) imLs_sz.x * (int) imLs_sz.y, // idist
        CUFFT_C2C, 
        (int) imLs_sz.z
    );
    cufftExecC2C(
        plan_many, 
        (cufftComplex *)&d_latentZ[0], 
        (cufftComplex *)&d_latentZ[0], 
        CUFFT_INVERSE
    );
    cufftDestroy(plan_many);
    
    ifftCorrection_sub<<<N_BLOCKS, N_THREADS>>>(d_latentZ, imLs_sz);

    plhs[0] = mxGPUCreateMxArrayOnGPU(latentZ);
    plhs[1] = mxGPUCreateMxArrayOnGPU(subwave);

    mxGPUDestroyGPUArray(latentZ);
    mxGPUDestroyGPUArray(subwave);
    // mxGPUDestroyGPUArray(ft_wavefront1); 
    mxGPUDestroyGPUArray(wavefront1);
    mxGPUDestroyGPUArray(pupil);
    mxGPUDestroyGPUArray(obseY);
    mxGPUDestroyGPUArray(ledIdx);
}