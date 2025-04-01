#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>


// 日志宏
#define LOG(level, msg) std::cout << "[" << #level << "] " << msg << std::endl
#define LOG_PROGRESS(iteration, total) \
    std::cout << "\rProgress: " << iteration << "/" << total << " (" \
              << static_cast<int>(100.0 * iteration / total) << "%)" << std::flush

// 检查CUDA错误的辅助函数
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// GPU间直接P2P传输实现
template<typename T>
void basic_alltoall_p2p(
    T** send_buffers,
    T** recv_buffers,
    int num_ranks,
    int rank,
    int chunk_size,
    int hidden_dim,
    cudaStream_t stream
) {
    const size_t segment_bytes = chunk_size * hidden_dim * sizeof(T);
    
    for (int dst_rank = 0; dst_rank < num_ranks; dst_rank++) {
        if (dst_rank == rank) {
            // 本地拷贝
            cudaMemcpyAsync(
                recv_buffers[rank] + dst_rank * chunk_size * hidden_dim,
                send_buffers[rank] + dst_rank * chunk_size * hidden_dim,
                segment_bytes, cudaMemcpyDeviceToDevice, stream);
        } else {
            // 跨设备拷贝
            cudaMemcpyPeerAsync(
                recv_buffers[dst_rank] + rank * chunk_size * hidden_dim,  // 目标地址+目标设备
                dst_rank,
                send_buffers[rank] + dst_rank * chunk_size * hidden_dim,  // 源地址+源设备
                rank,
                segment_bytes, stream);
        }
    }
}

int main(int argc, char* argv[]) {
    // 默认参数
    int num_ranks = 4;           // 使用4个GPU
    int chunk_size = 1024;       // 每个rank发送的数据块大小
    int hidden_dim = 1024;       // 隐藏层维度
    int num_iterations = 10;     // 性能测试迭代次数
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
    
    // 打印配置信息
    LOG(INFO, "Starting P2P AlltoAll test");
    LOG(INFO, "Configuration:");
    LOG(INFO, "  num_ranks   = " << num_ranks);
    LOG(INFO, "  chunk_size  = " << chunk_size);
    LOG(INFO, "  hidden_dim  = " << hidden_dim);
    LOG(INFO, "  iterations  = " << num_iterations);
    LOG(INFO, "  warmup      = " << warmup_iterations);
    
    // 检查可用GPU数量
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (num_ranks > device_count) {
        LOG(ERROR, "Requested " << num_ranks << " GPUs but only " 
                  << device_count << " available");
        return 1;
    }
    LOG(INFO, "Detected " << device_count << " CUDA-capable devices");
    
    // 分配设备内存
    using DataType = float;
    const size_t buffer_size = num_ranks * chunk_size * hidden_dim * sizeof(DataType);
    
    std::vector<DataType*> send_buffers(num_ranks);
    std::vector<DataType*> recv_buffers(num_ranks);
    std::vector<DataType*> host_buffers(num_ranks);
    std::vector<cudaStream_t> streams(num_ranks);
    
    // 初始化每个GPU
    LOG(INFO, "Initializing GPU resources...");
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaMalloc(&send_buffers[r], buffer_size));
        CHECK_CUDA(cudaMalloc(&recv_buffers[r], buffer_size));
        CHECK_CUDA(cudaMallocHost(&host_buffers[r], buffer_size));
        CHECK_CUDA(cudaStreamCreate(&streams[r]));
        
        // 初始化主机数据: rank ID作为基值
        for (int i = 0; i < num_ranks * chunk_size * hidden_dim; i++) {
            host_buffers[r][i] = static_cast<DataType>(r + i * 0.0001);
        }
        CHECK_CUDA(cudaMemcpy(send_buffers[r], host_buffers[r], buffer_size, 
                             cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(recv_buffers[r], 0, buffer_size));
        
        LOG(INFO, "  GPU " << r << " : " 
            << (buffer_size / (1024.0 * 1024.0)) << " MB allocated");
    }
    
    // 启用P2P访问
    LOG(INFO, "Enabling peer access...");
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        for (int p = 0; p < num_ranks; p++) {
            if (p != r) {
                int can_access = 0;
                CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, r, p));
                if (can_access) {
                    CHECK_CUDA(cudaDeviceEnablePeerAccess(p, 0));
                    LOG(INFO, "  GPU " << r << " -> GPU " << p << " : P2P enabled");
                } else {
                    LOG(WARNING, "  GPU " << r << " cannot access GPU " << p);
                }
            }
        }
    }
    
    // 预热运行
    LOG(INFO, "Warmup starts (" << warmup_iterations << " iterations)...");
    for (int i = 0; i < warmup_iterations; i++) {
        for (int r = 0; r < num_ranks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            basic_alltoall_p2p(send_buffers.data(), recv_buffers.data(),
                              num_ranks, r, chunk_size, hidden_dim, streams[r]);
        }
        // 同步所有设备
        for (int r = 0; r < num_ranks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            CHECK_CUDA(cudaStreamSynchronize(streams[r]));
        }
        LOG_PROGRESS(i+1, warmup_iterations);
    }
    std::cout << std::endl;
    
    // 性能测试
    LOG(INFO, "Performance test starts (" << num_iterations << " iterations)...");
    std::vector<double> timings;
    for (int iter = 0; iter < num_iterations; iter++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // 启动所有设备的传输
        for (int r = 0; r < num_ranks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            basic_alltoall_p2p(send_buffers.data(), recv_buffers.data(),
                              num_ranks, r, chunk_size, hidden_dim, streams[r]);
        }
        
        // 等待所有操作完成
        for (int r = 0; r < num_ranks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            CHECK_CUDA(cudaStreamSynchronize(streams[r]));
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        timings.push_back(elapsed.count());
        LOG_PROGRESS(iter+1, num_iterations);
    }
    std::cout << std::endl;
    
    // 统计结果
    double total_time = 0;
    double min_time = 1e9;
    double max_time = 0;
    for (const auto& t : timings) {
        total_time += t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }
    const double avg_time = total_time / num_iterations;
    const double throughput = (num_ranks * buffer_size / (1024.0 * 1024.0 * 1024.0)) 
                            / (avg_time / 1000.0);
    
    // LOG(INFO, "Performance results:");
    // LOG(INFO, "  Total data per iteration: " 
    //     << (num_ranks * buffer_size / (1024.0 * 1024.0)) << " MB");
    // LOG(INFO, "  Average time:   " << avg_time << " ms");
    // LOG(INFO, "  Min time:        " << min_time << " ms");
    // LOG(INFO, "  Max time:        " << max_time << " ms");
    // LOG(INFO, "  Aggregate throughput: " << throughput << " GB/s");
    
    std::cout << "\n\n--- Performance Summary ---\n";
    std::cout << "Configuration: " << num_ranks << " GPUs, "
              << "chunk_size=" << chunk_size 
              << ", hidden_dim=" << hidden_dim << "\n";
    std::cout << "Data size per GPU: "
              << (buffer_size / (1024.0 * 1024.0)) << " MB\n";

    std::cout << "basic_alltoall_p2p:    ";
    if (avg_time < 1.0) {
        std::cout << std::fixed << std::setprecision(7);
    } else {
        std::cout << std::fixed << std::setprecision(3);
    }
    std::cout << avg_time << " ms\n";

    // 数据验证（可选）
    LOG(INFO, "Validating data...");
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaMemcpy(host_buffers[r], recv_buffers[r], buffer_size, 
                             cudaMemcpyDeviceToHost));
        
        bool validation_passed = true;
        for (int src_rank = 0; src_rank < num_ranks; src_rank++) {
            const DataType expected = static_cast<DataType>(src_rank + 
                (r * chunk_size * hidden_dim) * 0.0001);
            const DataType actual = host_buffers[r][src_rank * chunk_size * hidden_dim];
            
            if (std::abs(actual - expected) > 1e-4) {
                LOG(ERROR, "  GPU " << r << " received invalid data from GPU " << src_rank 
                          << " at position 0: expected " << expected << ", got " << actual);
                validation_passed = false;
                break;
            }
        }
        if (validation_passed) {
            LOG(INFO, "  GPU " << r << " data validation passed");
        }
    }
    
    // 清理资源
    LOG(INFO, "Cleaning up resources...");
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaFree(send_buffers[r]));
        CHECK_CUDA(cudaFree(recv_buffers[r]));
        CHECK_CUDA(cudaFreeHost(host_buffers[r]));
        CHECK_CUDA(cudaStreamDestroy(streams[r]));
        // 禁用P2P访问
        for (int p = 0; p < num_ranks; p++) {
            if (p != r) cudaDeviceDisablePeerAccess(p);
        }
    }
    
    LOG(INFO, "Test completed successfully");
    return 0;
}