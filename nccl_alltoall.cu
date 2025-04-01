#include <cuda_runtime.h>
#include <nccl.h>
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

// NCCL版本的AlltoAll实现
template<typename T>
void nccl_alltoall(
    T* send_buffer,
    T* recv_buffer,
    int chunk_size,
    int hidden_dim,
    ncclDataType_t dtype,
    ncclComm_t comm,
    cudaStream_t stream
) {
    int rank, nranks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nranks);
    
    for (int r = 0; r < nranks; r++) {
        int send_offset = r * chunk_size * hidden_dim;
        int recv_offset = r * chunk_size * hidden_dim;
        
        ncclGroupStart();
        ncclSend(send_buffer + send_offset, chunk_size * hidden_dim, 
                 dtype, r, comm, stream);
        ncclRecv(recv_buffer + recv_offset, chunk_size * hidden_dim, 
                 dtype, r, comm, stream);
        ncclGroupEnd();
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

// 检查NCCL错误的辅助函数
#define CHECK_NCCL(call) { \
    ncclResult_t err = call; \
    if (err != ncclSuccess) { \
        std::cerr << "NCCL error in " << __FILE__ << " line " << __LINE__ << ": " \
                  << ncclGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int main(int argc, char* argv[]) {
    // 默认参数
    int num_ranks = 4;
    int chunk_size = 1024;
    int hidden_dim = 1024;
    int num_iterations = 10;
    int warmup_iterations = 3;
    
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

    LOG("INFO", "Starting NCCL AlltoAll test...");
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

    // 分配内存
    using DataType = float;
    ncclDataType_t nccl_dtype = ncclFloat;
    
    std::vector<DataType*> send_buffers(num_ranks);
    std::vector<DataType*> recv_buffers(num_ranks);
    std::vector<DataType*> send_buffers_host(num_ranks);
    std::vector<DataType*> recv_buffers_host(num_ranks);
    std::vector<cudaStream_t> streams(num_ranks);
    
    size_t buffer_size = num_ranks * chunk_size * hidden_dim * sizeof(DataType);

    LOG("INFO", "Initializing CUDA resources...");
    
    // 初始化每个GPU的资源
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
        
        CHECK_CUDA(cudaMemcpy(send_buffers[r], send_buffers_host[r], buffer_size, 
                             cudaMemcpyHostToDevice));
    }

    LOG("INFO", "Initializing NCCL...");
    
    // 初始化NCCL通信
    ncclUniqueId nccl_id;
    CHECK_NCCL(ncclGetUniqueId(&nccl_id));
    
    std::vector<ncclComm_t> comms(num_ranks);
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_NCCL(ncclCommInitRank(&comms[r], num_ranks, nccl_id, r));
    }

    LOG("INFO", "Starting warmup (" << warmup_iterations << " iterations)...");
    
    // 预热阶段
    for (int i = 0; i < warmup_iterations; i++) {
        for (int r = 0; r < num_ranks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            nccl_alltoall(send_buffers[r], recv_buffers[r], 
                         chunk_size, hidden_dim, nccl_dtype, comms[r], streams[r]);
        }
        
        for (int r = 0; r < num_ranks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            CHECK_CUDA(cudaStreamSynchronize(streams[r]));
        }
    }

    LOG("INFO", "Running performance test (" << num_iterations << " iterations)...");
    
    // 性能测试
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        LOG("DEBUG", "Iteration " << i+1 << "/" << num_iterations);
        
        for (int r = 0; r < num_ranks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            nccl_alltoall(send_buffers[r], recv_buffers[r], 
                         chunk_size, hidden_dim, nccl_dtype, comms[r], streams[r]);
        }
        
        for (int r = 0; r < num_ranks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            CHECK_CUDA(cudaStreamSynchronize(streams[r]));
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double avg_time = elapsed.count() / num_iterations;

    // 验证结果
    LOG("INFO", "Verifying results...");
    bool verification_passed = true;
    
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaMemcpy(recv_buffers_host[r], recv_buffers[r], buffer_size, 
                             cudaMemcpyDeviceToHost));
        
        // 验证逻辑（这里仅作示例）
        for (int i = 0; i < 10; i++) {
            if (std::abs(recv_buffers_host[r][i] - send_buffers_host[r][i]) > 1e-6) {
                verification_passed = false;
                LOG("ERROR", "Verification failed at GPU " << r << ", element " << i);
                break;
            }
        }
    }

    // 输出结果
    std::cout << "\n--- Performance Summary ---\n";
    std::cout << "Configuration: " << num_ranks << " GPUs, chunk_size=" << chunk_size 
              << ", hidden_dim=" << hidden_dim << "\n";
    std::cout << "Data size per GPU: " << (buffer_size / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "NCCL AlltoAll average time: " << avg_time << " ms\n";
    if (verification_passed) {
        LOG("INFO", "Result verification passed!");
    }

    // 清理资源
    LOG("INFO", "Cleaning up resources...");
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaFree(send_buffers[r]));
        CHECK_CUDA(cudaFree(recv_buffers[r]));
        CHECK_CUDA(cudaFreeHost(send_buffers_host[r]));
        CHECK_CUDA(cudaFreeHost(recv_buffers_host[r]));
        CHECK_CUDA(cudaStreamDestroy(streams[r]));
        ncclCommDestroy(comms[r]);
    }

    LOG("INFO", "Test completed successfully!");
    return 0;
}