#include <mpi.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <thread>

#include <mscclpp/core.hpp>

// 测量bootstrap初始化时间
void benchmark_bootstrap(int rank, int world_size, int iterations = 10) {
    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        // 同步所有进程
        MPI_Barrier(MPI_COMM_WORLD);

        // 开始计时
        auto start = std::chrono::high_resolution_clock::now();

        // 创建bootstrap
        auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world_size);
        mscclpp::UniqueId id;

        // 只在rank 0上创建UniqueId
        if (rank == 0) {
            id = bootstrap->createUniqueId();
        }

        // 广播UniqueId给所有进程
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

        // 初始化bootstrap
        bootstrap->initialize(id);

        // 确保所有进程都完成了初始化
        bootstrap->barrier();

        // 结束计时
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        // 收集所有进程的时间
        double local_time = elapsed.count();
        double max_time;
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            times.push_back(max_time);
            std::cout << "Iteration " << i << ": " << max_time << " ms" << std::endl;
        }

        // 等待一段时间，确保所有资源都被释放
        MPI_Barrier(MPI_COMM_WORLD);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 计算统计信息
    if (rank == 0 && !times.empty()) {
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double mean = sum / times.size();

        std::vector<double> diff(times.size());
        std::transform(times.begin(), times.end(), diff.begin(),
                       [mean](double x) { return x - mean; });

        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        double stddev = std::sqrt(sq_sum / times.size());

        double min_time = *std::min_element(times.begin(), times.end());
        double max_time = *std::max_element(times.begin(), times.end());

        // 排除第一次运行（通常会更慢）
        if (times.size() > 1) {
            double sum_without_first = std::accumulate(times.begin() + 1, times.end(), 0.0);
            double mean_without_first = sum_without_first / (times.size() - 1);

            std::cout << "\nBootstrap Performance Summary (" << world_size << " processes, excluding first run):" << std::endl;
            std::cout << "  Min time: " << min_time << " ms" << std::endl;
            std::cout << "  Max time: " << max_time << " ms" << std::endl;
            std::cout << "  Avg time: " << mean_without_first << " ms" << std::endl;
            std::cout << "  Stddev  : " << stddev << " ms" << std::endl;
        }

        std::cout << "\nBootstrap Performance Summary (" << world_size << " processes, all runs):" << std::endl;
        std::cout << "  Min time: " << min_time << " ms" << std::endl;
        std::cout << "  Max time: " << max_time << " ms" << std::endl;
        std::cout << "  Avg time: " << mean << " ms" << std::endl;
        std::cout << "  Stddev  : " << stddev << " ms" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // 初始化MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // 获取rank和world_size
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 解析命令行参数
    int iterations = 10;
    if (argc > 1) {
        iterations = std::stoi(argv[1]);
    }

    // 打印基本信息
    if (rank == 0) {
        std::cout << "Running bootstrap benchmark with " << world_size << " processes" << std::endl;
        std::cout << "Number of iterations: " << iterations << std::endl;
    }

    // 运行benchmark
    benchmark_bootstrap(rank, world_size, iterations);

    // 结束MPI
    MPI_Finalize();
    return 0;
}
