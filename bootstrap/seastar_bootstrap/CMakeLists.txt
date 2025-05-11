cmake_minimum_required(VERSION 3.10)
project(FastBootstrap CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

# Find MSCCLPP
set(MSCCLPP_INCLUDE_DIR "/root/kernel_dev/curr/ucxx-dev/.pixi/envs/default/include")
set(MSCCLPP_LIBRARY "/root/kernel_dev/curr/ucxx-dev/.pixi/envs/default/lib/libmscclpp.so")
include_directories(${MSCCLPP_INCLUDE_DIR})

# Add the executable
add_executable(bootstrap_benchmark
    bootstrap_benchmark.cc
    fast_bootstrap.cc
)

# Link against MPI and MSCCLPP
target_link_libraries(bootstrap_benchmark
    ${MPI_CXX_LIBRARIES}
    ${MSCCLPP_LIBRARY}
    pthread
)

# Add compiler flags
target_compile_options(bootstrap_benchmark PRIVATE
    -Wall
    -Wextra
    -O3
    -march=native
    -mtune=native
)

# Install the executable
install(TARGETS bootstrap_benchmark
    RUNTIME DESTINATION bin
)
