#!/bin/bash

# Settings
SRC=openMP-2.0.c
PROG=./openMP               # name of the compiled executable
OUTFILE="./results/weak-scaling.csv"    # CSV file
N0=65536                     # base problem size for p=1
K=512                      # number of layers
# CORES=$(grep -c ^processor /proc/cpuinfo) # number of (logical) cores
CORES=$(sysctl -n hw.logicalcpu)
NREPS=10                    # number of repetitions

# Compile
echo "Compiling $SRC ..."
clang -Wall -Wextra -Wpedantic -std=c99 -fopenmp -O3 -ffast-math -fuse-ld=lld -march=native "$SRC" -o "$PROG" -lm
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful."

# Write CSV header
echo "p$(seq -s , 1 $NREPS | sed 's/[0-9]\+/t&/g')" > "$OUTFILE"

# Collect times
for p in $(seq $CORES); do
    TIMES=()
    # scale N with p
    N=$((N0 * p))
    for rep in $(seq $NREPS); do
        EXEC_TIME=$(OMP_NUM_THREADS=$p "$PROG" $N $K | \
            grep "Execution time" | sed 's/Execution time //' | grep -o -E '[0-9]+\.[0-9]+')
        TIMES+=("$EXEC_TIME")
    done
    echo "$p,${TIMES[*]// /,}" >> "$OUTFILE"
done

echo "Results written to $OUTFILE"
