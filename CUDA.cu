#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <utility>
#include <errno.h>
#include <cuda_runtime.h>

#include "hpc.h"


#define R      3
#define BIAS   0.1f
#define BLKDIM 1024
#define RADIUS ((R-1)/2)


// Sigmoid activation function
__device__ static inline float sigmoid_device(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

void fillArrayWithRandom(float arr[], size_t size) {
    for (size_t i = 0; i < size; i++) {
        // rand() / RAND_MAX gives [0,1]
        // multiply by 2 → [0,2]
        // subtract 1 → [-1,1]
        arr[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
}

typedef struct {
    float *base;   /* original cudaMalloc pointer */
    float *input;
    float *Weights;
    float *output;
} Layer;

void initializeLayerOnDevice(Layer *layer, int N, size_t total_weights) {
    size_t size_input = N * sizeof(float);
    size_t size_W = total_weights * sizeof(float);
    size_t M = N - (R - 1); // Output size
    size_t size_output = M * sizeof(float);
    size_t total_size = size_input + size_W + size_output;

    cudaSafeCall(cudaMalloc((void**)&layer->base, total_size));

    layer->input = layer->base;
    layer->Weights = (float *)((char *)layer->base + size_input);
    layer->output = (float *)((char *)layer->base + size_input + size_W);
}

__global__ void forward(
    const float* x,
    const float* W,
    float* y,
    int out_size
) {
    const int neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_index >= out_size) return;
    float sum = BIAS;
    for (int offset = 0; offset < R; offset++) {
        sum += x[neuron_index + offset] * W[neuron_index * R + offset];
    }
    y[neuron_index] = sigmoid_device(sum);
}


__global__ void shared_forward(
    const float* __restrict__ x,
    const float* __restrict__ W,
    float* __restrict__ y,
    int out_size
) {
    __shared__ float x_shared[BLKDIM + 2 * RADIUS];
    int local_index = threadIdx.x;
    const int global_index = blockIdx.x * blockDim.x + local_index;

    // Load a chunk of x into shared memory
    // Assuming x is large enough, each thread loads one element
    if (global_index < out_size + 2*RADIUS) {
        x_shared[local_index] = x[global_index];
    }
    __syncthreads(); // make sure shared memory is populated

    if (global_index >= out_size) return;
    float sum = BIAS;
    for (int offset = 0; offset < R; offset++) {
        sum += x_shared[local_index + offset] * W[global_index * R + offset];
    }
    y[global_index] = sigmoid_device(sum);
}

__global__ void stencil1d(int *in, int *out) 
{
    __shared__ int temp[BLKDIM + 2 * RADIUS];
    const int gindex = threadIdx.x + blockIdx.x * blockDim.x + RADIUS;
    const int lindex = threadIdx.x + RADIUS;
    int result = 0, offset;
    /* Read input elements into shared memory */
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + blockDim.x] = in[gindex + blockDim.x];
    }
    __syncthreads(); 
    /* Apply the stencil */
    for (offset = -RADIUS ; offset <= RADIUS ; offset++) {
        result += temp[lindex + offset];
    }
    /* Store the result */
    out[gindex] = result;
}

int main(int argc, char *argv[]) {
    float *host_input;
    float *host_weights;
    Layer device_layer;
    Layer shared_device_layer;

    double start_time, stop_time, spent_time;
    double start_time_shared, stop_time_shared, spent_time_shared;


    // this line sets the random seed which in this case is the system's time
    srand(time(NULL));

    if (argc != 3) return fprintf(stderr, "Usage: %s <N> <K>\n", argv[0]), 1;

    char *end; errno = 0;
    long tempN = strtol(argv[1], &end, 10);
    if (errno || *end || tempN <= 0 || tempN > UINT_MAX) return fprintf(stderr, "Invalid N\n"), 1;
    unsigned int N = (unsigned int) tempN;

    long tempK = strtol(argv[2], &end, 10);
    if (errno || *end || tempK <= 1 || tempK > UINT_MAX) return fprintf(stderr, "Invalid K\n"), 1;
    unsigned int K = (unsigned int) tempK;

    // Calculate layer sizes: N - t*(R-1) for layer t
    int *layer_sizes = (int *) malloc(K * sizeof(int));
    for (int t = 0; t < K; t++) {
        layer_sizes[t] = N - t * (R - 1);
        printf("Layer %d: %d - %d*(%d-1) = %d neurons\n", t, N, t, R, layer_sizes[t]);
        // Validate that we don't have negative or zero neurons
        if (layer_sizes[t] <= 0) {
            printf("Error: Layer %d would have %d neurons (invalid)!\n", t, layer_sizes[t]);
            printf("Try reducing K or R, or increasing N.\n");
            exit(1);
        }
    }
    printf("Neurons in the first layer: %u,\nNumber of Layers: %u\n", N, K);

    host_input = (float*) malloc((N) * sizeof(float));
    fillArrayWithRandom(host_input, (size_t) N);

    // Calculate total weights needed for all layers
    size_t total_weights = 0;
    for (int t = 0; t < K-1; t++) {
        total_weights += layer_sizes[t+1] * R;  // Each layer needs output_size * R weights
    }
    host_weights = (float *)malloc(total_weights * sizeof(float));
    if (!host_weights) {
        fprintf(stderr, "Failed to allocate host memory for weights\n");
        return EXIT_FAILURE;
    }
    // Initialize all weights
    fillArrayWithRandom(host_weights, total_weights);

    unsigned int *weight_offsets = (unsigned int *)malloc((K-1) * sizeof(unsigned int));
    if (!weight_offsets) {
        fprintf(stderr, "Failed to allocate host memory for weight offsets\n");
        return EXIT_FAILURE;
    }
    for (int t = 0; t < K-1; t++) {
        // Calculate weight offset for layers
        size_t weight_offset = 0;
        for (int i = 0; i < t; i++) {
            weight_offset += layer_sizes[i+1] * R;  // Add weights used by previous layers
        }
        weight_offsets[t] = (unsigned int)weight_offset;
    }

    initializeLayerOnDevice(&device_layer, N, total_weights);
    printf("host input: \n");
    for (int i = 0; i < N && i < 10; i++) {
        printf("%.6f ", host_input[i]);
    }
    printf("\n");

    cudaSafeCall(cudaMemcpy(device_layer.input, host_input, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(device_layer.Weights, host_weights, total_weights * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(); 

    start_time = hpc_gettime();
    for (int t = 0; t < K-1; t++) {
        forward<<<(layer_sizes[t+1] + BLKDIM - 1) / BLKDIM, BLKDIM>>>(
            device_layer.input,
            device_layer.Weights + weight_offsets[t],  // Offset for current layer's weights
            device_layer.output,
            layer_sizes[t+1]
        );
        // cudaCheckError(); 
        // // Print intermediate results (all layers, not just K-2)
        // float *h_output = (float *)malloc(layer_sizes[t+1] * sizeof(float));
        // if (!h_output) { 
        //     fprintf(stderr, "Host malloc failed.\n"); 
        //     return EXIT_FAILURE; 
        // }
        // cudaSafeCall(cudaMemcpy(h_output, device_layer.output, layer_sizes[t+1] * sizeof(float), cudaMemcpyDeviceToHost));
        // printf("Output of layer %d (first 10 values or less):\n", t+1);  // Fixed: t+1, not t
        // for (int i = 0; i < layer_sizes[t+1] && i < 10; i++) {
        //     printf("%.6f ", h_output[i]);
        // }
        // printf("\n");
        
        // free(h_output);  // FIXED: Add this line to prevent memory leak
        
        // Swap buffers for next iteration
        if (t < K - 2) {  // Only swap if there's another iteration
            float *temp_ptr = device_layer.input;
            device_layer.input = device_layer.output;
            device_layer.output = temp_ptr;
        }
    }
    
    cudaCheckError();
    stop_time = hpc_gettime();
    spent_time = stop_time - start_time;
    printf("Total time for %u layers: %.6f seconds\n", K, spent_time);


    // FIXED: Copy from device_layer.output (where final result actually is)
    float *host_output = (float*) malloc(layer_sizes[K-1] * sizeof(float));
    if (!host_output) {
        fprintf(stderr, "Failed to allocate host memory for output\n");
        return EXIT_FAILURE;
    }
    
    // The final result is in device_layer.output after the last iteration
    cudaSafeCall(cudaMemcpy(host_output, device_layer.output, layer_sizes[K-1] * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n");
    printf("Final output (first 10 values or less):\n");
    for (int i = 0; i < layer_sizes[K-1] && i < 10; i++) {
        printf("%.6f ", host_output[i]);
    }
    printf("\n");

    // Now do the same with shared memory kernel
    initializeLayerOnDevice(&shared_device_layer, N, total_weights);
    printf("host input: \n");
        for (int i = 0; i < N && i < 10; i++) {
            printf("%.6f ", host_input[i]);
        }
        printf("\n");

    cudaSafeCall(cudaMemcpy(device_layer.input, host_input, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(device_layer.Weights, host_weights, total_weights * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError();
    start_time_shared = hpc_gettime();
    for (int t = 0; t < K-1; t++) {
        shared_forward<<<(layer_sizes[t+1] + BLKDIM - 1) / BLKDIM, BLKDIM>>>(
            device_layer.input,
            device_layer.Weights + weight_offsets[t],  // Offset for current layer's weights
            device_layer.output,
            layer_sizes[t+1]
        );
        // Swap buffers for next iteration
        if (t < K - 2) {  // Only swap if there's another iteration
            float *temp_ptr = device_layer.input;
            device_layer.input = device_layer.output;
            device_layer.output = temp_ptr;
        }
    }
    cudaCheckError();
    stop_time_shared = hpc_gettime();
    spent_time_shared = stop_time_shared - start_time_shared;
    printf("Total time for shared memory kernel: %.6f seconds\n", spent_time_shared);

   // FIXED: Copy from device_layer.output (where final result actually is)
    float *host_shared_output = (float*) malloc(layer_sizes[K-1] * sizeof(float));
    if (!host_shared_output) {
        fprintf(stderr, "Failed to allocate host memory for output\n");
        return EXIT_FAILURE;
    }
    
    // The final result is in device_layer.output after the last iteration
    cudaSafeCall(cudaMemcpy(host_shared_output, device_layer.output, layer_sizes[K-1] * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n");
    printf("Final output (first 10 values or less):\n");
    for (int i = 0; i < layer_sizes[K-1] && i < 10; i++) {
        printf("%.6f ", host_shared_output[i]);
    }
    printf("\n");

    // Clean up
    free(host_input);
    free(host_weights);
    free(host_output);
    free(layer_sizes);
    cudaFree(device_layer.base);
    
    return 0;
}