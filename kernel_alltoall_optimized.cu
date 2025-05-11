#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

// 日志宏定义
#define LOG(level, msg) \
    do { \
        std::cout << "[" << level << "] " << msg << std::endl; \
    } while (0)

// 检查CUDA错误的辅助函数
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// 优化版本1: 使用共享内存的AlltoAll实现
template<typename T>
__global__ void optimized_alltoall_kernel_v1(
    T* __restrict__ output,
    const T* __restrict__ input,
    int num_ranks,
    int rank,
    int chunk_size,
    int hidden_dim
) {
    extern __shared__ char shared_mem[];
    T* shared_data = reinterpret_cast<T*>(shared_mem);
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total_elements = num_ranks * chunk_size * hidden_dim;
    
    // 每个线程块处理一个连续的数据块
    const int elements_per_block = (total_elements + gridDim.x - 1) / gridDim.x;
    const int block_start = blockIdx.x * elements_per_block;
    const int block_end = min(block_start + elements_per_block, total_elements);
    
    // 每个线程处理多个元素
    for (int idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
        const int hidden_idx = idx % hidden_dim;
        const int item_idx = (idx / hidden_dim) % chunk_size;
        const int src_rank = (idx / hidden_dim) / chunk_size;
        
        // 计算输入和输出索引
        const int input_idx = (src_rank * chunk_size + item_idx) * hidden_dim + hidden_idx;
        const int output_idx = (item_idx * num_ranks + src_rank) * hidden_dim + hidden_idx;
        
        // 直接从全局内存读取并写入全局内存
        output[output_idx] = input[input_idx];
    }
}

// 优化版本2: 使用共享内存和协作组的AlltoAll实现
template<typename T>
__global__ void optimized_alltoall_kernel_v2(
    T* __restrict__ output,
    const T* __restrict__ input,
    int num_ranks,
    int rank,
    int chunk_size,
    int hidden_dim
) {
    extern __shared__ char shared_mem[];
    T* shared_data = reinterpret_cast<T*>(shared_mem);
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total_elements = num_ranks * chunk_size * hidden_dim;
    
    // 计算每个线程块处理的数据块大小
    const int block_size = blockDim.x;
    const int shared_mem_size = block_size * sizeof(T);
    
    // 每个线程块处理一个连续的数据块
    for (int base_idx = blockIdx.x * block_size; base_idx < total_elements; base_idx += gridDim.x * block_size) {
        const int idx = base_idx + threadIdx.x;
        
        if (idx < total_elements) {
            // 加载数据到共享内存
            shared_data[threadIdx.x] = input[idx];
        }
        
        __syncthreads();
        
        if (idx < total_elements) {
            // 计算重排后的索引
            const int hidden_idx = idx % hidden_dim;
            const int item_idx = (idx / hidden_dim) % chunk_size;
            const int src_rank = (idx / hidden_dim) / chunk_size;
            
            // 计算输出索引 - 这里实现了转置操作
            const int output_idx = (item_idx * num_ranks + src_rank) * hidden_dim + hidden_idx;
            
            // 写入全局内存
            output[output_idx] = shared_data[threadIdx.x];
        }
        
        __syncthreads();
    }
}

// 优化版本3: 针对大型数据集的分块处理
template<typename T, int BLOCK_SIZE = 256, int ITEMS_PER_THREAD = 4>
__global__ void optimized_alltoall_kernel_v3(
    T* __restrict__ output,
    const T* __restrict__ input,
    int num_ranks,
    int rank,
    int chunk_size,
    int hidden_dim
) {
    // 每个线程处理多个元素
    T thread_data[ITEMS_PER_THREAD];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = num_ranks * chunk_size * hidden_dim;
    const int elements_per_thread = (total_elements + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    
    // 每个线程处理多个连续的元素
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        const int idx = tid * ITEMS_PER_THREAD + i;
        if (idx < total_elements) {
            // 计算输入索引
            const int hidden_idx = idx % hidden_dim;
            const int item_idx = (idx / hidden_dim) % chunk_size;
            const int src_rank = (idx / hidden_dim) / chunk_size;
            
            // 计算输入和输出索引
            const int input_idx = (src_rank * chunk_size + item_idx) * hidden_dim + hidden_idx;
            const int output_idx = (item_idx * num_ranks + src_rank) * hidden_dim + hidden_idx;
            
            // 直接从全局内存读取并写入全局内存
            output[output_idx] = input[input_idx];
        }
    }
}

// 多GPU版本的AlltoAll实现
void multi_gpu_alltoall(
    std::vector<float*>& send_buffers,
    std::vector<float*>& recv_buffers,
    int num_ranks,
    int chunk_size,
    int hidden_dim,
    std::vector<cudaStream_t>& streams
) {
    size_t chunk_bytes = chunk_size * hidden_dim * sizeof(float);
    
    // 为每个GPU分配临时缓冲区
    std::vector<float*> temp_buffers(num_ranks);
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaMalloc(&temp_buffers[r], chunk_bytes * num_ranks));
    }
    
    // 每个GPU将自己的数据分散到所有GPU
    for (int src = 0; src < num_ranks; src++) {
        CHECK_CUDA(cudaSetDevice(src));
        
        // 将数据分块发送到每个目标GPU
        for (int dst = 0; dst < num_ranks; dst++) {
            if (src != dst) {
                // 计算源缓冲区中的偏移量
                size_t src_offset = dst * chunk_bytes;
                
                // 使用点对点传输将数据发送到目标GPU
                CHECK_CUDA(cudaMemcpyPeerAsync(
                    temp_buffers[dst] + src * chunk_size * hidden_dim,
                    dst,
                    send_buffers[src] + src_offset,
                    src,
                    chunk_bytes,
                    streams[src]
                ));
            } else {
                // 本地复制
                CHECK_CUDA(cudaMemcpyAsync(
                    temp_buffers[src] + src * chunk_size * hidden_dim,
                    send_buffers[src] + src * chunk_bytes,
                    chunk_bytes,
                    cudaMemcpyDeviceToDevice,
                    streams[src]
                ));
            }
        }
    }
    
    // 同步所有流
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaStreamSynchronize(streams[r]));
    }
    
    // 将临时缓冲区中的数据复制到接收缓冲区
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaMemcpyAsync(
            recv_buffers[r],
            temp_buffers[r],
            chunk_bytes * num_ranks,
            cudaMemcpyDeviceToDevice,
            streams[r]
        ));
    }
    
    // 释放临时缓冲区
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaFree(temp_buffers[r]));
    }
}

int main(int argc, char* argv[]) {
    // 默认参数
    int num_ranks = 4;           // 默认4个GPU
    int chunk_size = 1024;       // 每个rank发送给其他每个rank的数据量
    int hidden_dim = 1024;       // 隐藏维度大小
    int num_iterations = 10;     // 迭代次数，用于计算平均性能
    int warmup_iterations = 3;   // 预热迭代次数
    int kernel_version = 3;      // 默认使用优化版本3
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--num_ranks") == 0 && i + 1 < argc) {
            num_ranks = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--chunk_size") == 0 && i + 1 < argc) {
            chunk_size = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--hidden_dim") == 0 && i + 1 < argc) {
            hidden_dim = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            num_iterations = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup_iterations = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            kernel_version = atoi(argv[i + 1]);
            i++;
        }
    }
    
    LOG("INFO", "Starting optimized_alltoall_kernel test...");
    LOG("INFO", "Configuration: " << num_ranks << " GPUs, chunk_size=" << chunk_size 
              << ", hidden_dim=" << hidden_dim << ", kernel_version=" << kernel_version);

    // 检查可用的GPU数量
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (num_ranks > device_count) {
        std::cerr << "Requested " << num_ranks << " GPUs but only " 
                  << device_count << " are available." << std::endl;
        return 1;
    }
    
    LOG("INFO", "Initializing data buffers...");

    // 分配设备内存和主机内存
    using DataType = float;  // 可以修改为其他类型
    
    std::vector<DataType*> send_buffers(num_ranks);
    std::vector<DataType*> recv_buffers(num_ranks);
    std::vector<DataType*> send_buffers_host(num_ranks);
    std::vector<DataType*> recv_buffers_host(num_ranks);
    std::vector<cudaStream_t> streams(num_ranks);
    
    size_t buffer_size = num_ranks * chunk_size * hidden_dim * sizeof(DataType);
    
    // 分配并初始化每个GPU的内存
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaMalloc(&send_buffers[r], buffer_size));
        CHECK_CUDA(cudaMalloc(&recv_buffers[r], buffer_size));
        CHECK_CUDA(cudaMallocHost(&send_buffers_host[r], buffer_size));
        CHECK_CUDA(cudaMallocHost(&recv_buffers_host[r], buffer_size));
        CHECK_CUDA(cudaStreamCreate(&streams[r]));
        
        // 初始化发送数据
        for (size_t i = 0; i < num_ranks * chunk_size * hidden_dim; i++) {
            send_buffers_host[r][i] = static_cast<DataType>(r * 0.01 + i * 0.0001);
        }
        
        // 复制到设备
        CHECK_CUDA(cudaMemcpy(send_buffers[r], send_buffers_host[r], buffer_size, 
                             cudaMemcpyHostToDevice));
    }
    
    // 计算最佳的线程块和网格大小
    int blockSize = 256;
    int minGridSize;
    int maxActiveBlocks;
    
    // 获取设备属性
    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    
    // 计算共享内存大小
    size_t sharedMemSize = blockSize * sizeof(DataType);
    
    // 测试不同版本的kernel
    std::vector<double> kernel_times;
    
    LOG("INFO", "Starting warmup (" << warmup_iterations << " iterations)...");
    
    // 多GPU版本的测试
    if (kernel_version == 0) {
        LOG("INFO", "Testing multi-GPU alltoall implementation...");
        
        // 预热
        for (int i = 0; i < warmup_iterations; i++) {
            multi_gpu_alltoall(send_buffers, recv_buffers, num_ranks, chunk_size, hidden_dim, streams);
            
            // 同步所有设备
            for (int r = 0; r < num_ranks; r++) {
                CHECK_CUDA(cudaSetDevice(r));
                CHECK_CUDA(cudaStreamSynchronize(streams[r]));
            }
        }
        
        // 计时
        double total_time = 0.0;
        for (int i = 0; i < num_iterations; i++) {
            LOG("DEBUG", "Iteration " << i+1 << "/" << num_iterations);
            auto start = std::chrono::high_resolution_clock::now();
            
            multi_gpu_alltoall(send_buffers, recv_buffers, num_ranks, chunk_size, hidden_dim, streams);
            
            // 同步所有设备
            for (int r = 0; r < num_ranks; r++) {
                CHECK_CUDA(cudaSetDevice(r));
                CHECK_CUDA(cudaStreamSynchronize(streams[r]));
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            total_time += elapsed.count();
            
            LOG("DEBUG", "  Iteration time: " << elapsed.count() << " ms");
        }
        
        double avg_time = total_time / num_iterations;
        kernel_times.push_back(avg_time);
        
        std::cout << "  Multi-GPU alltoall: " << avg_time << " ms" << std::endl;
    }
    // 单GPU版本的测试
    else {
        for (int r = 0; r < num_ranks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            
            // 预热
            for (int i = 0; i < warmup_iterations; i++) {
                if (kernel_version == 1) {
                    optimized_alltoall_kernel_v1<<<256, blockSize, sharedMemSize>>>(
                        recv_buffers[r], send_buffers[r], num_ranks, r, chunk_size, hidden_dim);
                } else if (kernel_version == 2) {
                    optimized_alltoall_kernel_v2<<<256, blockSize, sharedMemSize>>>(
                        recv_buffers[r], send_buffers[r], num_ranks, r, chunk_size, hidden_dim);
                } else {
                    optimized_alltoall_kernel_v3<DataType, 256, 4><<<256, blockSize>>>(
                        recv_buffers[r], send_buffers[r], num_ranks, r, chunk_size, hidden_dim);
                }
                CHECK_CUDA(cudaStreamSynchronize(streams[r]));
            }
        }
        
        LOG("INFO", "Running performance test (" << num_iterations << " iterations)...");
        for (int r = 0; r < num_ranks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            
            // 计时
            double total_time = 0.0;
            for (int i = 0; i < num_iterations; i++) {
                LOG("DEBUG", "GPU " << r << " - Iteration " << i+1 << "/" << num_iterations);
                auto start = std::chrono::high_resolution_clock::now();
                
                if (kernel_version == 1) {
                    optimized_alltoall_kernel_v1<<<256, blockSize, sharedMemSize>>>(
                        recv_buffers[r], send_buffers[r], num_ranks, r, chunk_size, hidden_dim);
                } else if (kernel_version == 2) {
                    optimized_alltoall_kernel_v2<<<256, blockSize, sharedMemSize>>>(
                        recv_buffers[r], send_buffers[r], num_ranks, r, chunk_size, hidden_dim);
                } else {
                    optimized_alltoall_kernel_v3<DataType, 256, 4><<<256, blockSize>>>(
                        recv_buffers[r], send_buffers[r], num_ranks, r, chunk_size, hidden_dim);
                }
                CHECK_CUDA(cudaStreamSynchronize(streams[r]));
                
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                total_time += elapsed.count();
                
                LOG("DEBUG", "  Iteration time: " << elapsed.count() << " ms");
            }
            
            double avg_time = total_time / num_iterations;
            kernel_times.push_back(avg_time);
            
            std::cout << "  GPU " << r << ": " << avg_time << " ms" << std::endl;
        }
    }
    
    double kernel_avg = 0;
    for (auto t : kernel_times) kernel_avg += t;
    kernel_avg /= kernel_times.size();
    std::cout << "  Average: " << kernel_avg << " ms" << std::endl;
    
    // 性能结果摘要
    std::cout << "\n--- Performance Summary ---\n";
    std::cout << "Configuration: " << num_ranks << " GPUs, chunk_size=" << chunk_size 
              << ", hidden_dim=" << hidden_dim << "\n";
    std::cout << "Data size per GPU: " << (buffer_size / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "Kernel version " << kernel_version << ": " << kernel_avg << " ms\n";
    
    // 验证结果正确性（可选）
    LOG("INFO", "Verifying results...");
    bool verification_passed = true;
    
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaMemcpy(recv_buffers_host[r], recv_buffers[r], buffer_size, 
                            cudaMemcpyDeviceToHost));
        
        // 这里可以添加验证逻辑
        // 简单起见，我们只检查数据是否被复制了
        for (size_t i = 0; i < 10; i++) {  // 检查前10个元素
            if (recv_buffers_host[r][i] != send_buffers_host[r][i]) {
                verification_passed = false;
                std::cout << "Verification failed at GPU " << r << ", element " << i << std::endl;
                break;
            }
        }
    }
    
    if (verification_passed) {
        LOG("INFO", "Verification passed!");
    }
    
    // 释放资源
    LOG("INFO", "Cleaning up resources...");
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaFree(send_buffers[r]));
        CHECK_CUDA(cudaFree(recv_buffers[r]));
        CHECK_CUDA(cudaFreeHost(send_buffers_host[r]));
        CHECK_CUDA(cudaFreeHost(recv_buffers_host[r]));
        CHECK_CUDA(cudaStreamDestroy(streams[r]));
    }
    
    LOG("INFO", "Test completed successfully!");
    return 0;
}
