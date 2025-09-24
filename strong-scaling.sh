#!/bin/bash

# Settings
SRC=openMP.c
PROG=./openMP              # name of the compiled executable
OUTFILE=strong-scaling.csv # CSV file
N=1000000                  # problem size;
K=1024                     # number of layers 2^10
CORES=$(grep -c ^processor /proc/cpuinfo) # number of logical cores
NREPS=10                   # number of repetitions

# Compile
echo "Compiling $SRC ..."
gcc -Wall -Wpedantic -std=c99 -fopenmp -O3 -ffast-math -fuse-ld=lld -march=native "$SRC" -o "$PROG"
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
    for rep in $(seq $NREPS); do
        EXEC_TIME=$(OMP_NUM_THREADS=$p "$PROG" $N $K | \
            grep "Execution time" | sed 's/Execution time //' | grep -o -E '[0-9]+\.[0-9]+')
        TIMES+=("$EXEC_TIME")
    done
    echo "$p,${TIMES[*]// /,}" >> "$OUTFILE"
done

echo "Results written to $OUTFILE"

