# Implicit Neural Representation of a 2D image using Hash-Nerf. (Implementation in MATLAB)

This MATLAB + CUDA implementation replicates the hash encoding technique used in [Instant-NGP (Instant Neural Graphics Primitives)](https://nvlabs.github.io/instant-ngp/), specifically focusing on the multi-resolution hash encoding component.

## Overview

Hash encoding is a compact way to represent high-dimensional data (like 3D coordinates) using spatial hashing and multi-resolution grids. This implementation provides:

- Multi-resolution grid initialization
- Spatial hashing with gradient computation
- Hash table lookups with linear interpolation
- Example usage for encoding 3D coordinates

## Installation & Usage
run InstantNGP_fit_image_2025.m
