#include <stdio.h>     // printf, fprintf
#include <stdlib.h>    // malloc, free, rand, srand, strtol, exit
#include <math.h>      // expf
#include <omp.h>       // OpenMP pragmas + omp_get_wtime
#include <time.h>      // time
#include <errno.h>     // errno
#include <limits.h>   // UINT_MAX

#include "hpc.h"

const float BIAS = 0.1;
const int R = 3;

// Neural network structure
typedef struct {
    int N;              // Number of neurons in input layer
    int K;              // Total number of layers (including input and output)
    int *layer_sizes;   // Array storing the number of neurons in each layer (array of size K)
    float **neurons;    // 2D array: neurons[layer][neuron_index] stores neuron values (2D array of size K x layer_sizes[t])
    float ***weights;   // 3D array: weights[layer][neuron][connection] stores connection weights (3D array of size (K-1) x layer_sizes[t+1] x R)
                        // layer_sizes[t+1] = N-(R-1)
} NeuralNetwork;

// Sigmoid activation function
static inline float sigmoid(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Initialize the neural network
static inline NeuralNetwork* init_network(int N, int K) {
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    nn->N = N;
    nn->K = K;
    // nn->R = R;
    
    // Calculate layer sizes: N - t*(R-1) for layer t
    nn->layer_sizes = malloc(K * sizeof(int));
    for (int t = 0; t < K; t++) {
        nn->layer_sizes[t] = N - t * (R - 1);
        printf("Layer %d: %d - %d*(%d-1) = %d neurons\n", 
        t, N, t, R, nn->layer_sizes[t]);
        
        // Validate that we don't have negative or zero neurons
        if (nn->layer_sizes[t] <= 0) {
            printf("Error: Layer %d would have %d neurons (invalid)!\n", t, nn->layer_sizes[t]);
            printf("Try reducing K or R, or increasing N.\n");
            exit(1);
        }
    }
    
    // Allocate memory for neurons
    nn->neurons = malloc(K * sizeof(float*));
    for (int i = 0; i < K; i++) {
        nn->neurons[i] = malloc(nn->layer_sizes[i] * sizeof(float));
    }
    
    // Allocate memory for weights (K-1 weight matrices between K layers)
    nn->weights = malloc((K-1) * sizeof(float**));
    for (int i = 0; i < K-1; i++) {
        nn->weights[i] = malloc(nn->layer_sizes[i+1] * sizeof(float*));
        for (int j = 0; j < nn->layer_sizes[i+1]; j++) {
            nn->weights[i][j] = malloc(R * sizeof(float));
        }
    }
    
    // Initialize weights randomly between -1 and 1
    for (int i = 0; i < K-1; i++) {
        for (int j = 0; j < nn->layer_sizes[i+1]; j++) {
            for (int k = 0; k < R; k++) {
                nn->weights[i][j][k] = ((float)rand() / (double)RAND_MAX) * 2.0f - 1.0f;
            }
        }
    }
    
    return nn;
}

// Initialize input layer with random values
void init_input(NeuralNetwork *nn) {
    for (int i = 0; i < nn->N; i++) {
        nn->neurons[0][i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
}

void forward_pass(NeuralNetwork *nn) {
    /*
     * Create a parallel region once outside the main loop. This avoids the
     * expensive overhead of creating and destroying threads for each layer.
     *
     * default(none): A best practice that forces us to explicitly state the
     * sharing attribute for all variables from the outer scope, preventing
     * accidental data races.
     *
     * shared(nn): The pointer to the network struct is shared. All threads
     * see and modify the single, original network object in memory. This is
     * essential for them to collaborate on the same data.
     */
    #pragma omp parallel default(none) shared(nn)
    {
        // This outer loop over layers remains sequential, as the calculation of
        // layer 't' depends entirely on the completed results of layer 't-1'.
        // However, the body of the loop is executed by the entire team of threads.
        for (int layer = 1; layer < nn->K; layer++) {
            int current_layer_size = nn->layer_sizes[layer];
            int prev_layer_size = nn->layer_sizes[layer - 1];

            /*
             * This pragma divides the iterations of the following 'for' loop
             * among the threads in the team. Each thread calculates a subset
             * of the neurons for the current layer.
             *
             * schedule(static): We use a static schedule because the workload
             * for each neuron is identical. This minimizes scheduling overhead
             * by giving each thread a single, large chunk of work.
             */
            #pragma omp for schedule(static)
            for (int i = 0; i < current_layer_size; i++) {
                float sum = BIAS;
                
                for (int j = 0; j < R; j++) {
                    int prev_idx = i + j;
                    // The 'if' condition makes automatic vectorization harder.
                    // For this to be truly effective, we would need to ensure
                    // there are no out-of-bounds accesses by other means.
                    // However, since R is very small, the performance impact is minor.
                    if (prev_idx < prev_layer_size) {
                        sum += nn->neurons[layer-1][prev_idx] *
                               nn->weights[layer-1][i][j];
                    }
                }
                // Apply sigmoid activation
                nn->neurons[layer][i] = sigmoid(sum);
            }
            // An implicit barrier exists at the end of an 'omp for' loop.
            // All threads will wait here, ensuring the entire layer is computed
            // before any thread proceeds to the next 'layer' iteration. This
            // synchronization is critical for the algorithm's correctness.
        }
    } // All threads are joined here, after all layers are processed.
}

// Print network architecture
void print_architecture(NeuralNetwork *nn) {
    printf("\nNeural Network Architecture:\n");
    printf("Input neurons (N): %d\n", nn->N);
    printf("Total layers (K): %d\n", nn->K);
    printf("Consecutive dependency (R): %d\n", R);
    printf("Layer sizes: ");
    int total_neurons = 0;
    for (int i = 0; i < nn->K; i++) {
        printf("%d", nn->layer_sizes[i]);
        total_neurons += nn->layer_sizes[i];
        if (i < nn->K - 1) printf(" -> ");
    }
    printf("\n");
    printf("Total neurons in network: %d\n", total_neurons);
}

// Print network state
void print_network(NeuralNetwork *nn) {
    printf("\nNetwork State:\n");
    for (int i = 0; i < nn->K; i++) {
        printf("Layer %d: ", i);
        for (int j = 0; j < nn->layer_sizes[i]; j++) {
            if (j == 10) { printf("... "); break; } // Limit output for large layers
            printf("%.4f ", nn->neurons[i][j]);
        }
        printf("\n");
    }
}

// Print weights (optional, for debugging)
void print_weights(NeuralNetwork *nn) {
    printf("\nWeights:\n");
    for (int i = 0; i < nn->K - 1; i++) {
        printf("Layer %d to %d:\n", i, i+1);
        for (int j = 0; j < nn->layer_sizes[i+1]; j++) {
            printf("  Neuron %d: ", j);
            for (int k = 0; k < R; k++) {
                printf("%.4f ", nn->weights[i][j][k]);
            }
            printf("\n");
        }
    }
}

// Free memory
void free_network(NeuralNetwork *nn) {
    // Free neurons
    for (int i = 0; i < nn->K; i++) {
        free(nn->neurons[i]);
    }
    free(nn->neurons);
    
    // Free weights
    for (int i = 0; i < nn->K-1; i++) {
        for (int j = 0; j < nn->layer_sizes[i+1]; j++) {
            free(nn->weights[i][j]);
        }
        free(nn->weights[i]);
    }
    free(nn->weights);
    
    free(nn->layer_sizes);
    free(nn);
}

int main(int argc, char *argv[]) {
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

    printf("Neurons in the first layer: %u,\nNumber of Layers: %u\n", N, K);
    
    printf("Initializing Feed-Forward Neural Network with OpenMP...\n");
    
    // Initialize network
    NeuralNetwork *nn = init_network(N, K);
    
    // Print architecture
    print_architecture(nn);
    
    // Initialize input
    init_input(nn);
    
    // Perform forward pass
    printf("\nPerforming forward pass...\n");
    double start_time = hpc_gettime();
    forward_pass(nn);
    double end_time = hpc_gettime();

    // Print results
    print_network(nn);
    
    printf("\nForward pass completed in %.6f seconds\n", end_time - start_time);
    printf("Final output layer has %d neurons\n", nn->layer_sizes[nn->K-1]);
    
    // Optionally print weights for small networks (only for demonstration)
    if (N <= 6) {
        print_weights(nn);
    }
    
    // Clean up
    free_network(nn);
    
    return 0;
}