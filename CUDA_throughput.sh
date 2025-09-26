#!/bin/bash

# Usage: ./run_experiments.sh source_file.cu fixed_N fixed_K
# Example: ./run_experiments.sh my_kernel.cu 1024 64

SRC=CUDA.cu
FIXED_N=1048576
FIXED_K=1024
PROG="./CUDA"

# --- Compile ---
echo "Compiling $SRC ..."
nvcc -g -G "$SRC" -o "$PROG" \
  -gencode arch=compute_50,code=sm_50 \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful."

# --- Make results folder ---
mkdir -p results

# --- CSV output files ---
CSV_FIXED_N="results/fixed_N_${FIXED_N}.csv"
CSV_FIXED_K="results/fixed_K_${FIXED_K}.csv"

echo "K,no_shared_time,no_shared_throughput,shared_time,shared_throughput" > $CSV_FIXED_N
echo "N,no_shared_time,no_shared_throughput,shared_time,shared_throughput" > $CSV_FIXED_K

# --- Sweep over K values (fixed N) ---
for K in 2 4 8 16 32 64 128 256 512 1024; do
    OUTPUT=$($PROG $FIXED_N $K)

    no_shared_time=$(echo "$OUTPUT" | grep "No shared memory execution time" | awk -F': ' '{print $2}' | xargs)
    no_shared_tp=$(echo "$OUTPUT" | grep "Throughput no shared memory" | awk -F': ' '{print $2}' | xargs)
    shared_time=$(echo "$OUTPUT" | grep "Shared memory execution time" | awk -F': ' '{print $2}' | xargs)
    shared_tp=$(echo "$OUTPUT" | grep "Throughput with shared memory" | awk -F': ' '{print $2}' | xargs)

    echo "$K,$no_shared_time,$no_shared_tp,$shared_time,$shared_tp" >> $CSV_FIXED_N
done

# --- Sweep over N values (fixed K) ---
for N in 4096 8192 16384 32768 65536 131072 262144 524288 1048576; do
    OUTPUT=$($PROG $N $FIXED_K)

    no_shared_time=$(echo "$OUTPUT" | grep "No shared memory execution time" | awk -F': ' '{print $2}' | xargs)
    no_shared_tp=$(echo "$OUTPUT" | grep "Throughput no shared memory" | awk -F': ' '{print $2}' | xargs)
    shared_time=$(echo "$OUTPUT" | grep "Shared memory execution time" | awk -F': ' '{print $2}' | xargs)
    shared_tp=$(echo "$OUTPUT" | grep "Throughput with shared memory" | awk -F': ' '{print $2}' | xargs)

    echo "$N,$no_shared_time,$no_shared_tp,$shared_time,$shared_tp" >> $CSV_FIXED_K
done

echo "✅ Done. Results saved in:"
echo "   - $CSV_FIXED_N"
echo "   - $CSV_FIXED_K"
