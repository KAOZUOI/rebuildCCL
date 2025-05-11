#!/bin/bash

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
make -j$(nproc)

# Return to the original directory
cd ..

echo "Build completed. Run the benchmark with:"
echo "mpirun -n 8 ./build/bootstrap_benchmark [iterations]"
