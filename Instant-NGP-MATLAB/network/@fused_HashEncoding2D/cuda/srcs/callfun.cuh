#pragma once

#include "addon/mxGPUArray.h"
#include "kernels.cuh"
#include "mex.h"

void Call_Fwd(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const * prhs[]
);

void Call_Bwd(
    int nlhs, mxArray *plhs[], 
    int nrhs, mxArray const * prhs[]
);