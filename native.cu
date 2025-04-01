#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

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

// GPU间通信版本的AlltoAll实现 - 使用CUDA设备间直接P2P传输
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
    for (int dst_rank = 0; dst_rank < num_ranks; dst_rank++) {
        if (dst_rank == rank) {
            size_t bytes = chunk_size * hidden_dim * sizeof(T);
            cudaMemcpyAsync(
                recv_buffers[rank] + dst_rank * chunk_size * hidden_dim,
                send_buffers[rank] + dst_rank * chunk_size * hidden_dim,
                bytes, cudaMemcpyDeviceToDevice, stream);
        } else {
            size_t bytes = chunk_size * hidden_dim * sizeof(T);
            cudaMemcpyPeerAsync(
                recv_buffers[dst_rank] + rank * chunk_size * hidden_dim,
                dst_rank,
                send_buffers[rank] + dst_rank * chunk_size * hidden_dim,
                rank,
                bytes, stream);
        }
    }
}

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
    
    // 检查可用的GPU数量
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (num_ranks > device_count) {
        std::cerr << "Requested " << num_ranks << " GPUs but only " 
                  << device_count << " are available." << std::endl;
        return 1;
    }
    
    // 分配设备内存和主机内存
    using DataType = float;  // 可以修改为其他类型
    ncclDataType_t nccl_dtype = ncclFloat;
    
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
    
    // 为每个设备启用P2P访问
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        for (int p = 0; p < num_ranks; p++) {
            if (p != r) {
                int can_access = 0;
                CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, r, p));
                if (can_access) {
                    cudaDeviceEnablePeerAccess(p, 0);
                    cudaGetLastError(); // 清除可能的错误
                }
            }
        }
    }
    
    // 初始化NCCL
    ncclComm_t comm;
    int* dev_ranks = new int[num_ranks];
    for (int r = 0; r < num_ranks; r++) dev_ranks[r] = r;
    
    ncclUniqueId nccl_id;
    CHECK_NCCL(ncclGetUniqueId(&nccl_id));
    
    std::vector<ncclComm_t> comms(num_ranks);
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_NCCL(ncclCommInitRank(&comms[r], num_ranks, nccl_id, r));
    }
    
    // ---------------------------------------------------------------------
    // 测试1: basic_alltoall_kernel (单GPU内部数据重排)
    // ---------------------------------------------------------------------
    std::cout << "\nTesting basic_alltoall_kernel (single-GPU)...\n";
    std::vector<double> kernel_times;
    
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        
        // 预热
        for (int i = 0; i < warmup_iterations; i++) {
            basic_alltoall_kernel<<<256, 256>>>(
                recv_buffers[r], send_buffers[r], num_ranks, r, chunk_size, hidden_dim);
            CHECK_CUDA(cudaStreamSynchronize(streams[r]));
        }
        
        // 计时
        double total_time = 0.0;
        for (int i = 0; i < num_iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            basic_alltoall_kernel<<<256, 256>>>(
                recv_buffers[r], send_buffers[r], num_ranks, r, chunk_size, hidden_dim);
            CHECK_CUDA(cudaStreamSynchronize(streams[r]));
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            total_time += elapsed.count();
        }
        
        double avg_time = total_time / num_iterations;
        kernel_times.push_back(avg_time);
        
        std::cout << "  GPU " << r << ": " << avg_time << " ms" << std::endl;
    }
    
    double kernel_avg = 0;
    for (auto t : kernel_times) kernel_avg += t;
    kernel_avg /= num_ranks;
    std::cout << "  Average: " << kernel_avg << " ms" << std::endl;
    
    // ---------------------------------------------------------------------
    // 测试2: basic_alltoall_p2p (GPU间直接P2P通信)
    // ---------------------------------------------------------------------
    std::cout << "\nTesting basic_alltoall_p2p (P2P)...\n";
    
    // 清零接收缓冲区
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaMemset(recv_buffers[r], 0, buffer_size));
    }
    
    // 预热
    for (int i = 0; i < warmup_iterations; i++) {
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
    }
    
    // 计时
    auto p2p_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
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
    }
    
    auto p2p_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> p2p_elapsed = p2p_end - p2p_start;
    double p2p_avg = p2p_elapsed.count() / num_iterations;
    
    std::cout << "  Average time: " << p2p_avg << " ms" << std::endl;
    
    // ---------------------------------------------------------------------
    // 测试3: nccl_alltoall (使用NCCL库)
    // ---------------------------------------------------------------------
    std::cout << "\nTesting nccl_alltoall (NCCL)...\n";
    
    // 清零接收缓冲区
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaMemset(recv_buffers[r], 0, buffer_size));
    }
    
    // 预热
    for (int i = 0; i < warmup_iterations; i++) {
        for (int r = 0; r < num_ranks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            nccl_alltoall(send_buffers[r], recv_buffers[r], 
                         chunk_size, hidden_dim, nccl_dtype, comms[r], streams[r]);
        }
        
        // 等待所有操作完成
        for (int r = 0; r < num_ranks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            CHECK_CUDA(cudaStreamSynchronize(streams[r]));
        }
    }
    
    // 计时
    auto nccl_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        for (int r = 0; r < num_ranks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            nccl_alltoall(send_buffers[r], recv_buffers[r], 
                         chunk_size, hidden_dim, nccl_dtype, comms[r], streams[r]);
        }
        
        // 等待所有操作完成
        for (int r = 0; r < num_ranks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            CHECK_CUDA(cudaStreamSynchronize(streams[r]));
        }
    }
    
    auto nccl_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> nccl_elapsed = nccl_end - nccl_start;
    double nccl_avg = nccl_elapsed.count() / num_iterations;
    
    std::cout << "  Average time: " << nccl_avg << " ms" << std::endl;
    
    // 性能结果摘要
    std::cout << "\n--- Performance Summary ---\n";
    std::cout << "Configuration: " << num_ranks << " GPUs, chunk_size=" << chunk_size 
              << ", hidden_dim=" << hidden_dim << "\n";
    std::cout << "Data size per GPU: " << (buffer_size / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "basic_alltoall_kernel: " << kernel_avg << " ms\n";
    std::cout << "basic_alltoall_p2p:    " << p2p_avg << " ms\n";
    std::cout << "nccl_alltoall:         " << nccl_avg << " ms\n";
    
    // 释放资源
    for (int r = 0; r < num_ranks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaFree(send_buffers[r]));
        CHECK_CUDA(cudaFree(recv_buffers[r]));
        CHECK_CUDA(cudaFreeHost(send_buffers_host[r]));
        CHECK_CUDA(cudaFreeHost(recv_buffers_host[r]));
        CHECK_CUDA(cudaStreamDestroy(streams[r]));
        ncclCommDestroy(comms[r]);
    }
    
    delete[] dev_ranks;
    return 0;
}