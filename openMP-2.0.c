#include <stdio.h>     // printf, fprintf
#include <stdlib.h>    // malloc, free, rand, srand, strtol, exit
#include <math.h>      // expf
#include <omp.h>       // OpenMP pragmas + omp_get_wtime
#include <time.h>      // time
#include <errno.h>     // errno
#include <limits.h>   // UINT_MAX

#include "hpc.h"

unsigned int N = 0; // Number of neurons in input layer
unsigned int K = 0; // Total number of layers (including input and output)
const float BIAS = 0.1;
const int R = 3;

// Sigmoid activation function
static inline float sigmoid(const float x) {
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

int main(int argc, char *argv[]) {
    float *input, *output, *weights;
    int *layer_sizes;
    double start_time, stop_time;

    // this line sets the random seed which in this case is the system's time
    srand(time(NULL));

    if (argc != 3) return fprintf(stderr, "Usage: %s <N> <K>\n", argv[0]), 1;

    char *end; errno = 0;
    long tempN = strtol(argv[1], &end, 10);
    if (errno || *end || tempN <= 0 || tempN > UINT_MAX) return fprintf(stderr, "Invalid N\n"), 1;
    N = (unsigned int) tempN;

    long tempK = strtol(argv[2], &end, 10);
    if (errno || *end || tempK <= 1 || tempK > UINT_MAX) return fprintf(stderr, "Invalid K\n"), 1;
    K = (unsigned int) tempK;

    input = (float*) malloc((N) * sizeof(float));
    if (!input) {
        fprintf(stderr, "Failed to allocate host memory for input\n");
        return EXIT_FAILURE;
    }
    fillArrayWithRandom(input, (size_t) N);

    // Calculate layer sizes: N - t*(R-1) for layer t
    layer_sizes = (int *) malloc(K * sizeof(int));
    for (int t = 0; t < K; t++) {
        layer_sizes[t] = N - t * (R - 1);
        // Validate that we don't have negative or zero neurons
        if (layer_sizes[t] <= 0) {
            printf("Error: Layer %d would have %d neurons (invalid)!\n", t, layer_sizes[t]);
            printf("Try reducing K or R, or increasing N.\n");
            exit(1);
        }
    }

    output = (float *)malloc(layer_sizes[1] * sizeof(float));
    if (!output) {
        fprintf(stderr, "Failed to allocate host memory for output\n");
        return EXIT_FAILURE;
    }

    // Calculate total weights needed for all layers
    size_t total_weights = 0;
    for (int t = 0; t < K-1; t++) {
        total_weights += layer_sizes[t+1] * R;  // Each layer needs output_size * R weights
    }
    weights = (float *)malloc(total_weights * sizeof(float));
    if (!weights) {
        fprintf(stderr, "Failed to allocate host memory for weights\n");
        return EXIT_FAILURE;
    }
    // Initialize all weights
    fillArrayWithRandom(weights, total_weights);

    // Perform forward pass
    printf("\nPerforming forward pass...\n");

    float *weights_ptr = weights;
    int i;
    float sum;
    start_time = hpc_gettime();
    for (int layer = 1; layer < K; layer++) {
        int current_layer_size = layer_sizes[layer];
        int prev_layer_size = layer_sizes[layer - 1];
        #pragma omp parallel for schedule(static) default(none) \
                private(i, sum) \
                shared(input, output, weights_ptr, current_layer_size, prev_layer_size, BIAS)
        for (int i = 0; i < current_layer_size; i++) {
            float sum = BIAS;
            #pragma omp simd
            for (int offset = 0; offset < R; offset++) {
                int prev_idx = i + offset;
                if (prev_idx < prev_layer_size) {
                    sum += input[prev_idx] * weights_ptr[i * R + offset];
                }
            }
            output[i] = sigmoid(sum);
        }
        weights_ptr += layer_sizes[layer] * R;  // Move to the weights for the current layer
        // Swap the input and output arrays if we are not at the last layer
        if (layer < K - 1) {
            float *temp = input;
            input = output;
            output = temp;
        }
    }
    stop_time = hpc_gettime();

    printf("\nForward pass completed in %.6f seconds\n", stop_time - start_time);

    // Cleanup
    free(input);
    free(weights);
    free(output);
    free(layer_sizes);
    
    return 0;
}