<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Neural Pupil Engineering for off-axis Fourier Ptychography](#neural-pupil-engineering-for-off-axis-fourier-ptychography)
   * [News](#news)
   * [How to use ?](#how-to-use-)
      + [Neural Pupil Engineering FPM (NePE-FPM) Reconstruction](#neural-pupil-engineering-fpm-nepe-fpm-reconstruction)
      + [Fitting a single image using MATLAB-NGP](#fitting-a-single-image-using-matlab-ngp)
   * [Key Features](#key-features)
   * [How It Works](#how-it-works)
   * [Comparison](#comparison)
   * [License and Citation](#license-and-citation)

<!-- TOC end -->

<!-- TOC --><a name="neural-pupil-engineering-for-off-axis-fourier-ptychography"></a>
# Neural Pupil Engineering for off-axis Fourier Ptychography
<br>
This is the MATLAB code for the implementation of neural pupil engineering FPM (NePE-FPM), an optimization framework for FPM reconstruction for off-axis areas. <br>
<br>
NePE-FPM engineers the pupil function using an implicit neural representation with multi-resolution hash encoding, enabling continuous, smooth shifting of the pupil function without introducing additional physical parameters. <br>
<br>
By optimizing a feature-domain loss function, NePE-FPM adaptively filters Fourier-space information from low-resolution measurements, achieving accurate off-axis reconstruction without modeling off-axis propagation.<br>
<br>

<!-- TOC --><a name="news"></a>
## News
- **2025/09/10:**  :sparkles: Our paper has been accepted by _**Optica**_! <br>
- **2024/07/28:** 🔥 We released our MATLAB codes! <br>
<br>

<!-- TOC --><a name="how-to-use-"></a>
## How to use ?
<!-- TOC --><a name="neural-pupil-engineering-fpm-nepe-fpm-reconstruction"></a>
### Neural Pupil Engineering FPM (NePE-FPM) Reconstruction

This implementation provides both conventional and neural-enhanced reconstruction methods for Fourier Ptychographic Microscopy:

**Available Methods:**

- **Conventional FPM with Feature-Domain Loss**: 
  Execute `cuFPM_recon_FDFPM.m` for traditional FPM reconstruction utilizing [feature-domain](https://doi.org/10.1002/advs.202413975) optimization criteria.

- **Neural Pupil Engineering FPM (NePE-FPM)**:
  Run `cuFPM_recon_NePE.m` to perform reconstruction with our novel neural-enhanced pupil engineering approach.

<div align = 'center'>
<img src = "https://github.com/THUHoloLab/Neural-Pupil-Engineering-FPM/blob/main/resources/testNePE.gif" width = "800" alt="" align = center />
<br>
<em>NePF-FPM Reconstruction: Left - sample, Right - pupil function</em>
</div>

**Code Availability:**
Example implementations and sample datasets are accessible in the [NePE-FPM directory](https://github.com/THUHoloLab/Neural-Pupil-Engineering-FPM/tree/main/NePE-FPM).
<br>

<!-- TOC --><a name="fitting-a-single-image-using-matlab-ngp"></a>
### Fitting a single image using MATLAB-NGP

For researchers interested in neural graphics primitives implementation, we provide **MATLAB-NGP** - a MATLAB adaptation of NVIDIA's Instant Neural Graphics Primitives ([Instant-NGP](https://github.com/NVlabs/instant-ngp)). 

This implementation enables efficient single image reconstruction through coordinate-based neural representations:

**Key Features:**
- Multi-resolution hash encoding for compact feature representation
- GPU-accelerated inference and training
- Modular architecture compatible with MATLAB's deep learning framework

**Implementation Details:**
The complete MATLAB-NGP framework is available in the [Instant-NGP-MATLAB directory](https://github.com/THUHoloLab/Neural-Pupil-Engineering-FPM/tree/main/Instant-NGP-MATLAB), including example usage and pre-trained models for rapid deployment.

<div align = 'center'>
<img src = "https://github.com/THUHoloLab/Neural-Pupil-Engineering-FPM/blob/main/resources/test_hash32_4_2048.gif" width = "800" alt="Comparison of MLP-based vs Instant-NGP reconstruction" align = center />
<br>
<em>Comparative Reconstruction Performance: Left - Traditional MLP-based approach, Right - Instant-NGP accelerated reconstruction</em>
</div>
<br>


<!-- TOC --><a name="key-features"></a>
## Key Features

- **Dual Learning Strategy**: Simultaneously optimizes neural hash encodings and physical wavefront parameters
- **Hybrid Architecture**: MATLAB for control flow with CUDA-accelerated kernels for efficient computation  
- **Differentiable Physics**: End-to-end differentiable pipeline enabling gradient-based optimization
- **Off-axis Reconstruction**: Reconstruction of off-axis FOV using FPM can be achieved with high-speed, and high-quality. 
<br>

<!-- TOC --><a name="how-it-works"></a>
## How It Works

The system employs two parallel learning streams:
1. **Neural Path**: Learnable hash tables provide efficient coordinate-based representations
2. **Physical Path**: Optimizable sample wavefronts capture physical light propagation

Both streams converge through a differentiable FPM forward model, enabling joint optimization of neural representations and optical parameters for high-quality phase retrieval and image reconstruction.

The system follows a hybrid MATLAB-CUDA architecture for efficient forward and backward computations.

<div align = 'center'>
<img src = "https://github.com/THUHoloLab/Neural-Pupil-Engineering-FPM/blob/main/resources/flow_chart.png" width = "800" alt="" align = center />
</div><br>
<br>

<!-- TOC --><a name="comparison"></a>
## Comparison
We test NePE-FPM on a published dataset aims to address the phase curvature [[Codes](https://opticapublishing.figshare.com/articles/journal_contribution/Supplementary_document_for_Addressing_phase-curvature_in_Fourier_ptychography_-_5810780_pdf/19762861?file=52755638)] [[paper](https://doi.org/10.1364/OE.458657)]. The results are compared against other FPM algorithms. 

<div align = 'center'>
<img src = "https://github.com/THUHoloLab/Neural-Pupil-Engineering-FPM/blob/main/resources/public_data.png" width = "1000" alt="" align = center />
</div><br>

We test NePE-FPM on a our priviate [[dataset](https://github.com/THUHoloLab/Neural-Pupil-Engineering-FPM/tree/main/NePE-FPM)], where a edge-FOV is cropped for reconstruction. The results are compared against FD-FPM algorithms. 
<div align = 'center'>
<img src = "https://github.com/THUHoloLab/Neural-Pupil-Engineering-FPM/blob/main/resources/results.jpg" width = "1000" alt="" align = center />
</div><br>


<!-- TOC --><a name="license-and-citation"></a>
## License and Citation

This framework is licensed under the MIT License. Please see `LICENSE` for details.

If you use it in your research, we would appreciate a citation via
```bibtex
@article{zhang2025whole,
  title={Whole-field, high-resolution Fourier ptychography with neural pupil engineering},
  author={Zhang, Shuhe and Cao, Liangcai},
  journal={Optica},
  volume={},
  number={},
  pages={},
  year={2025},
  publisher={Optica Publishing Group},
  doi={10.1364/OPTICA.575065}
}
```
