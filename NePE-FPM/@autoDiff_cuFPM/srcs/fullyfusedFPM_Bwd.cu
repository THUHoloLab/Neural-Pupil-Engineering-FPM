#include "mex.h"
#include "cuda/mxGPUArray.h"
#include "cuda/kernels.cuh"
#include <cuda_runtime.h>

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
    // input parames
    const mxGPUArray_t * dl_doutput;
    const mxGPUArray_t * wavefront1;
    const mxGPUArray_t * wavefront2;
    const mxGPUArray_t * subwave;
    const mxGPUArray_t * ledIdx;

    // output parames
    mxGPUArray_t * dldw1;
    mxGPUArray_t * dldw2;
    mxGPUArray_t * latentZ;

    const real32_t * d_dl_doutput;
    const creal32_t * d_wavefront1;
    const creal32_t * d_wavefront2;
    const creal32_t * d_subwave;
    const int2 * d_ledIdx;

    mxInitGPU();

    dl_doutput  = mxGPUCreateFromMxArray(prhs[0]); // sample wavefront
    wavefront1  = mxGPUCreateFromMxArray(prhs[1]); // sample wavefront
    wavefront2  = mxGPUCreateFromMxArray(prhs[2]); // pupil wavefront
    latentZ     = mxGPUCopyFromMxArray(prhs[3]); // observed intensity
    subwave     = mxGPUCreateFromMxArray(prhs[4]); // observed intensity
    ledIdx      = mxGPUCreateFromMxArray(prhs[5]); // LED position in pixel

    d_dl_doutput  = getGPUDataRO<real32_t>(dl_doutput);
    d_wavefront1  = getGPUDataRO<creal32_t>(wavefront1);
    d_wavefront2  = getGPUDataRO<creal32_t>(wavefront2);
    d_subwave     = getGPUDataRO<creal32_t>(subwave);
    d_ledIdx      = getGPUDataRO<int2>(ledIdx);
    // d_pratio     = (const int *) mxGPUGetDataReadOnly(pratio);
    
    creal32_t * d_latentZ = (creal32_t * __restrict__ ) mxGPUGetData(latentZ);

    dldw1 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(wavefront1), 
        mxGPUGetDimensions(wavefront1),
        mxSINGLE_CLASS, 
        mxCOMPLEX, 
        MX_GPU_INITIALIZE_VALUES
    );

    dldw2 = mxGPUCreateGPUArray(
        mxGPUGetNumberOfDimensions(wavefront2), 
        mxGPUGetDimensions(wavefront2),
        mxSINGLE_CLASS, 
        mxCOMPLEX, 
        MX_GPU_INITIALIZE_VALUES
    );

    creal32_t * d_dldw1 = (creal32_t * __restrict__ ) mxGPUGetData(dldw1);
    creal32_t * d_dldw2 = (creal32_t * __restrict__ ) mxGPUGetData(dldw2);
    // creal32_t * d_white = (creal32_t * __restrict__ ) mxGPUGetData(white);
    // get image size                            
    dim3 imLs_sz = size2dim3(subwave);
    dim3 imHs_sz = size2dim3(wavefront1);

    dim3 N_BLOCKS = {
        (unsigned) (imLs_sz.x + BLOCK_X - 1) / BLOCK_X,
        (unsigned) (imLs_sz.y + BLOCK_Y - 1) / BLOCK_Y,
        (unsigned) imLs_sz.z
    };
    dim3 N_THREADS = {BLOCK_X,BLOCK_Y,1};

    backward_Z<<<N_BLOCKS, N_THREADS>>>(imLs_sz, d_dl_doutput, d_latentZ);
    // backwardLatentZ(d_latentZ, imLs_sz);
    int inembed[2];
    for (int i{0}; i < 2; i++) {
        inembed[i] = (int) imLs_sz.x;
    }
    cufftHandle plan_many;
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
    cufftExecC2C(plan_many, (cufftComplex *)d_latentZ, (cufftComplex *)d_latentZ, CUFFT_FORWARD);
    cufftDestroy(plan_many);
    cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(d_latentZ, (unsigned) imLs_sz.x);
    // fftshift done!
    fused_deconvPIE<<<N_BLOCKS, N_THREADS>>>(
        d_latentZ,
        d_wavefront2,
        d_subwave,
        d_ledIdx,
        imLs_sz, imHs_sz,
        d_dldw1, d_dldw2
    );
    // ifft2 d_dldw1
    N_BLOCKS = {
        (unsigned) ((imHs_sz.x + BLOCK_X - 1) / BLOCK_X),
        (unsigned) ((imHs_sz.y + BLOCK_Y - 1) / BLOCK_Y),
        (unsigned) 1
    };
    cufftShift_2D_kernel<<<N_BLOCKS, N_THREADS>>>(d_dldw1,(int) imHs_sz.x);

    cufftHandle plan;
    cufftPlan2d(&plan, imHs_sz.x, imHs_sz.y, CUFFT_C2C);
    cufftExecC2C(
        plan, 
        (cufftComplex *)d_dldw1, 
        (cufftComplex *)d_dldw1,
        CUFFT_INVERSE
    );
    cufftDestroy(plan);

    ifftCorrection<<<N_BLOCKS, N_THREADS>>>(d_dldw1,imHs_sz);

    plhs[0] = mxGPUCreateMxArrayOnGPU(dldw1);
    plhs[1] = mxGPUCreateMxArrayOnGPU(dldw2);

    mxGPUDestroyGPUArray(dldw1);
    mxGPUDestroyGPUArray(dldw2);

    mxGPUDestroyGPUArray(dl_doutput);
    mxGPUDestroyGPUArray(wavefront1);
    mxGPUDestroyGPUArray(wavefront2);
    mxGPUDestroyGPUArray(latentZ);
    mxGPUDestroyGPUArray(subwave);
    mxGPUDestroyGPUArray(ledIdx);
}