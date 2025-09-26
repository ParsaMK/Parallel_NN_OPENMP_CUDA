# Parallel_NN_OPENMP_CUDA

A parallel implementation of a neural network consisting of 1D stencil operations, optimized for both CPU (OpenMP) and GPU (CUDA) execution.

## Overview

This project provides two implementations of a parallel neural network:
- **OpenMP**: CPU-based parallelization using OpenMP
- **CUDA**: GPU-based parallelization using NVIDIA CUDA

## Compilation

### OpenMP (CPU)

To compile the OpenMP implementation:

```bash
gcc -Wall -Wextra -Wpedantic -std=c99 -fopenmp -O3 -ffast-math openMP-2.0 -o openMP-2.0 -lm
```

### CUDA (GPU)

To compile the CUDA implementation:

```bash
nvcc -g -G CUDA.cu -o CUDA \
  -gencode arch=compute_50,code=sm_50 \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90
```

The compilation flags ensure compatibility with GPU architectures from Maxwell through Hopper generations.

## Usage

Both executables accept the same command-line arguments:

### OpenMP
```bash
./openMP-2.0 [NUMBER-OF-NEURONS] [NUMBER-OF-LAYERS]
```

### CUDA
```bash
./CUDA [NUMBER-OF-NEURONS] [NUMBER-OF-LAYERS]
```

### Parameters

- `NUMBER-OF-NEURONS`: The number of neurons in the first layer
- `NUMBER-OF-LAYERS`: The total number of layers in the neural network

## Requirements

### OpenMP Version
- GCC compiler with OpenMP support
- C99 standard library

### CUDA Version
- NVIDIA CUDA Toolkit
- Compatible NVIDIA GPU (Maxwell architecture or newer)
- NVCC compiler

## GPU Architecture Support

The CUDA implementation supports the following GPU architectures:
- Maxwell (compute capability 5.0)
- Pascal (compute capability 6.1)
- Turing (compute capability 7.5)
- Ampere (compute capability 8.0, 8.9)
- Hopper (compute capability 9.0)

## Performance Notes

The implementation uses 1D stencil operations for efficient parallel computation of neural network layers. Both versions are optimized for their respective hardware platforms to maximize performance.