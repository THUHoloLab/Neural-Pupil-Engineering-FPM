# Neural Pupil Engineering for off-axis Fourier Ptychography
<br>
This is the MATLAB code for the implementation of neural pupil engineering FPM (NePE-FPM), an optimization framework for FPM reconstruction for off-axis areas. <br>
<br>
NePE-FPM engineers the pupil function using an implicit neural representation with multi-resolution hash encoding, enabling continuous, smooth shifting of the pupil function without introducing additional physical parameters. <br>
<br>
By optimizing a feature-domain loss function, NePE-FPM adaptively filters Fourier-space information from low-resolution measurements, achieving accurate off-axis reconstruction without modeling off-axis propagation.<br>
<br>

## News
- **2025/09/10:**  :sparkles: Our paper has been accepted by _**Optica**_! <br>
- **2024/07/28:** 🔥 We released our MATLAB codes! <br>

## Key Features

- **Dual Learning Strategy**: Simultaneously optimizes neural hash encodings and physical wavefront parameters
- **Hybrid Architecture**: MATLAB for control flow with CUDA-accelerated kernels for efficient computation  
- **Differentiable Physics**: End-to-end differentiable pipeline enabling gradient-based optimization
- **InstantNGP Integration**: Adapts multi-resolution hash encoding for computational microscopy

## How It Works

The system employs two parallel learning streams:
1. **Neural Path**: Learnable hash tables provide efficient coordinate-based representations
2. **Physical Path**: Optimizable sample wavefronts capture physical light propagation

Both streams converge through a differentiable FPM forward model, enabling joint optimization of neural representations and optical parameters for high-quality phase retrieval and image reconstruction.

The system follows a hybrid MATLAB-CUDA architecture for efficient forward and backward computations.

<div align = 'center'>
<img src = "https://github.com/THUHoloLab/Neural-Pupil-Engineering-FPM/blob/main/resources/flow_chart.png" width = "500" alt="" align = center />
</div><br>
