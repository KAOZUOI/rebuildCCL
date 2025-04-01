#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <string>

// 日志宏定义
#define LOG(level, msg) \
    do { \
        std::cout << "[" << level << "] " << msg << std::endl; \
    } while (0)

// 基础版本的intranode AlltoAll实现
template<typename T>
__global__ void basic_alltoall_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    int num_ranks,
    int rank,
    int chunk_size,
    int hidden_dim
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    const int total_elements = num_ranks * chunk_size * hidden_dim;
    
    for (int idx = tid; idx < total_elements; idx += stride) {
        const int hidden_idx = idx % hidden_dim;
        const int item_idx = (idx / hidden_dim) % chunk_size;
        const int src_rank = (idx / hidden_dim) / chunk_size;
        
        const int input_idx = (src_rank * chunk_size + item_idx) * hidden_dim + hidden_idx;
        const int output_idx = (src_rank * chunk_size + item_idx) * hidden_dim + hidden_idx;
        
        output[output_idx] = input[input_idx];
    }
}

// 检查CUDA错误的辅助函数
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int main(int argc, char* argv[]) {
    // 默认参数
    int num_ranks = 4;           // 默认4个GPU
    int chunk_size = 1024;       // 每个rank发送给其他每个rank的数据量
    int hidden_dim = 1024;       // 隐藏维度大小
    int num_iterations = 10;     // 迭代次数，用于计算平均性能
    int warmup_iterations = 3;   // 预热迭代次数
    
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
        }
    }
    
    LOG("INFO", "Starting basic_alltoall_kernel test...");
    LOG("INFO", "Configuration: " << num_ranks << " GPUs, chunk_size=" << chunk_size 
              << ", hidden_dim=" << hidden_dim);

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
    
    // ---------------------------------------------------------------------
    // 测试: basic_alltoall_kernel (单GPU内部数据重排)
    // ---------------------------------------------------------------------
    std::cout << "\nTesting basic_alltoall_kernel (single-GPU)...\n";
    std::vector<double> kernel_times;
    
    LOG("INFO", "Starting warmup (" << warmup_iterations << " iterations)...");
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        
        // 预热
        for (int i = 0; i < warmup_iterations; i++) {
            basic_alltoall_kernel<<<256, 256>>>(
                recv_buffers[r], send_buffers[r], num_ranks, r, chunk_size, hidden_dim);
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
            
            basic_alltoall_kernel<<<256, 256>>>(
                recv_buffers[r], send_buffers[r], num_ranks, r, chunk_size, hidden_dim);
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
    
    double kernel_avg = 0;
    for (auto t : kernel_times) kernel_avg += t;
    kernel_avg /= num_ranks;
    std::cout << "  Average: " << kernel_avg << " ms" << std::endl;
    
    // 性能结果摘要
    std::cout << "\n--- Performance Summary ---\n";
    std::cout << "Configuration: " << num_ranks << " GPUs, chunk_size=" << chunk_size 
              << ", hidden_dim=" << hidden_dim << "\n";
    std::cout << "Data size per GPU: " << (buffer_size / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "basic_alltoall_kernel: " << kernel_avg << " ms\n";
    
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