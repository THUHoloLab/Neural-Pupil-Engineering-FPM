#include "addon.h"
#include "fftshift_kernel.cuh"
#include <cooperative_groups.h>
#include <iostream>

__device__ float absC(const creal32_T in, const float ratio){
    float out;
    out = sqrtf(in.re * in.re + in.im * in.im) / ratio;
    return out;
}

__device__ float sign(const float in){
    float out;
    out = copysignf(1.0f,in);
    return out;
}

namespace cg = cooperative_groups;

__global__ void getSubpupil(
    const creal32_t* __restrict__ wavefront1,
    const creal32_t* __restrict__ wavefront2,
    const int2* __restrict__ ledindex,
    const dim3 imgSz,
    const dim3 imgSzL,
    creal32_t* __restrict__ subwave,
    creal32_t* __restrict__ latentZ
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    unsigned idz = block.group_index().z; 

    const bool inside = (idx < imgSz.x) && (idy < imgSz.y) && (idz < imgSz.z);

    unsigned pixL_x = (unsigned) ledindex[idz].x;
    unsigned pixL_y = (unsigned) ledindex[idz].y;

    __shared__ creal32_t this_pupil[BLOCK_SIZE];
    unsigned tr = block.thread_rank();
    unsigned pix_id = idx * imgSz.y + idy;
    this_pupil[tr] = wavefront2[pix_id];
    block.sync();

    if (inside) {
        unsigned pix_id = idx * imgSz.y + idy;
        unsigned page_id = idz * (imgSz.x * imgSz.y);
        unsigned pix_id_large = (idx + pixL_x - 1) * imgSzL.y + (idy + pixL_y - 1);

        creal32_t temp = wavefront1[pix_id_large];
        subwave[pix_id + page_id] = temp;

        creal32_t this_pupil0 = this_pupil[tr];
        float c = this_pupil0.re;
        float d = this_pupil0.im;

        creal32_t temp_latentZ;
        temp_latentZ.re = (temp.re * c - temp.im * d);
        temp_latentZ.im = (temp.re * d + c * temp.im);
        latentZ[pix_id + page_id] = temp_latentZ;
    }
}

__global__ void fused_deconvPIE(
    const creal32_t* __restrict__ recordZ,
    const creal32_t* __restrict__ pupil,
    const creal32_t* __restrict__ subwave,
    const int2* __restrict__ ledindex,
    const dim3 imLs_sz,
    const dim3 imHs_sz,
    creal32_t* __restrict__ dldw1,
    creal32_t* __restrict__ dldw2
){
    auto block = cg::this_thread_block();
    unsigned idx = block.group_index().x * block.group_dim().x + block.thread_index().x;
    unsigned idy = block.group_index().y * block.group_dim().y + block.thread_index().y; 
    unsigned idz = block.group_index().z; 
    // unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; 
    // unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
    // unsigned idz = blockIdx.z;   

    __shared__ creal32_t this_pupil[BLOCK_SIZE];
    unsigned tr = block.thread_rank();
    unsigned pix_id = idx * imLs_sz.y + idy;
    this_pupil[tr] = pupil[pix_id];
    block.sync();

    const bool inside = (idx < imLs_sz.x) && (idy < imLs_sz.y) && (idz < imLs_sz.z);  
    if (inside){
        unsigned idxZ = idz * (imLs_sz.x * imLs_sz.y);
        unsigned pix_id = idx * imLs_sz.y + idy;
        
        creal32_t tempX = recordZ[pix_id + idxZ];
        creal32_t tempP = this_pupil[tr];

        float a = tempX.re;
        float b = tempX.im;

        float c = tempP.re;
        float d = tempP.im;

        tempX.re = a * c + b * d;
        tempX.im = c * b - a * d;

        c = subwave[pix_id + idxZ].re;
        d = subwave[pix_id + idxZ].im;

        float pupil_re = a * c + b * d;
        float pupil_im = c * b - a * d;
        float *temp = (float *) dldw2;
        atomicAdd(temp + 2 * pix_id + 0, pupil_re);
        atomicAdd(temp + 2 * pix_id + 1, pupil_im);

        // stitch the Fourier spectrum
        unsigned pixL_x = (unsigned) ledindex[idz].x;
        unsigned pixL_y = (unsigned) ledindex[idz].y;
        const bool inside_FP = ((idx + pixL_x - 1) < imHs_sz.x) && ((idy + pixL_y - 1) < imHs_sz.y);
        if(inside_FP){
            float *temp_large = (float *) dldw1;    
            unsigned pix_id_large = (idx + pixL_x - 1) * imHs_sz.y + (idy + pixL_y - 1); 
            atomicAdd(temp_large + pix_id_large * 2 + 0, tempX.re);
            atomicAdd(temp_large + pix_id_large * 2 + 1, tempX.im);
        }
    }
}

__global__ void ifftCorrection(
    creal32_t* __restrict__ spectrum,
    const dim3 imHs_sz
){
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z;  

    const bool inside = (idx < imHs_sz.x) && (idy < imHs_sz.y);    
    if (inside){
        unsigned pix_id = idx * imHs_sz.y + idy;

        float ratio = 1 / ((float) (imHs_sz.x * imHs_sz.y));
        float a = spectrum[pix_id].re;
        float b = spectrum[pix_id].im;
        spectrum[pix_id].re = a * ratio;
        spectrum[pix_id].im = b * ratio;
    }
}

__global__ void setConstraint(
    const dim3 imLs_sz,
    const int pratio,
    const real32_t* __restrict__ img_Y,
    const creal32_t* __restrict__ latentz,
    creal32_t* __restrict__ out
){
    auto block = cg::this_thread_block();
    unsigned tr_x = block.thread_index().x;
    unsigned tr_y = block.thread_index().y;

    unsigned idx = block.group_index().x * block.group_dim().x + tr_x;
    unsigned idy = block.group_index().y * block.group_dim().y + tr_y; 
    // unsigned idz = block.group_index().z; 

    const bool inside = (idx < imLs_sz.x) && (idy < imLs_sz.y) && (block.group_index().z < imLs_sz.z);

    unsigned pix_A = idx * imLs_sz.y + idy;
    unsigned pix_X = (idx < (imLs_sz.x - 1))? ((idx + 1) * imLs_sz.y + idy) : idy;
    unsigned pix_Y = (idy < (imLs_sz.y - 1))? (idx * imLs_sz.y + idy + 1) : (idx * imLs_sz.y);
    unsigned page_id = block.group_index().z * (imLs_sz.x * imLs_sz.y);

    __shared__ float blk_x[BLOCK_X + 1][BLOCK_Y];
    __shared__ float blk_y[BLOCK_X][BLOCK_Y + 1];

    creal32_t lat_A = latentz[pix_A + page_id];

    float ratio = (float) (imLs_sz.x * imLs_sz.y * pratio * pratio);

    float tempB = img_Y[pix_A + page_id];
    float tempA = absC(lat_A,ratio) - tempB;

    blk_x[tr_x + 1][tr_y] = sign(absC(latentz[pix_X + page_id],ratio) - img_Y[pix_X + page_id] - tempA);
    blk_y[tr_x][tr_y + 1] = sign(absC(latentz[pix_Y + page_id],ratio) - img_Y[pix_Y + page_id] - tempA);

    block.sync();

    unsigned test_p2 = 0;
    if (tr_x == 0){
        test_p2 = (idx == 0)? ((imLs_sz.x - 1) * imLs_sz.y + idy) : ((idx - 1) * imLs_sz.y + idy);
        tempB = absC(latentz[test_p2 + page_id],ratio) - img_Y[test_p2 + page_id];
        blk_x[0][tr_y] = sign(tempA - tempB);
    }

    if (tr_y == 0){
        test_p2 = (idy == 0)? (idx * imLs_sz.y + imLs_sz.y - 1) : (idx * imLs_sz.y + idy - 1);
        tempB = absC(latentz[test_p2 + page_id],ratio) - img_Y[test_p2 + page_id];
        blk_y[tr_x][0] = sign(tempA - tempB);
    }

    block.sync();
    
    tempA = (blk_x[tr_x][tr_y] - blk_x[tr_x + 1][tr_y] + 
             blk_y[tr_x][tr_y] - blk_y[tr_x][tr_y + 1]) * ((float) pratio * pratio);

    float ang = atan2f(lat_A.im,lat_A.re);

    if (inside){
        out[pix_A + page_id].re = cosf(ang) * tempA;
        out[pix_A + page_id].im = sinf(ang) * tempA;
    }
}

__global__ void backward_Z(
    const dim3 imLs_sz,
    const real32_t* __restrict__ dl_doutput,
    creal32_t* __restrict__ latentz
){
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned idy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z;  
    // unsigned idz = block.group_index().z; 
    const bool inside = (idx < imLs_sz.x) && (idy < imLs_sz.y) && (idz < imLs_sz.z);

    if (inside){
        unsigned pix_A = idx * imLs_sz.y + idy;
        unsigned page_id = idz * (imLs_sz.x * imLs_sz.y);

        creal32_t lat_A = latentz[pix_A + page_id];
        
        float ang = atan2f(lat_A.im,lat_A.re);
        float tempA = dl_doutput[pix_A + page_id];

        lat_A.re = cosf(ang) * tempA;
        lat_A.im = sinf(ang) * tempA;

        latentz[pix_A + page_id] = lat_A;
    }
}