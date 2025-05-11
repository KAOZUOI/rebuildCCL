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
#include <fstream>
#include <iomanip>

#include <mscclpp/core.hpp>
#include "fast_bootstrap.h"

// 测量mscclpp bootstrap初始化时间
void benchmark_mscclpp_bootstrap(int rank, int world_size, int iterations = 10) {
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
            std::cout << "MSCCLPP Iteration " << i << ": " << max_time << " ms" << std::endl;
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

            std::cout << "\nMSCCLPP Bootstrap Performance Summary (" << world_size << " processes, excluding first run):" << std::endl;
            std::cout << "  Min time: " << min_time << " ms" << std::endl;
            std::cout << "  Max time: " << max_time << " ms" << std::endl;
            std::cout << "  Avg time: " << mean_without_first << " ms" << std::endl;
            std::cout << "  Stddev  : " << stddev << " ms" << std::endl;
        }

        std::cout << "\nMSCCLPP Bootstrap Performance Summary (" << world_size << " processes, all runs):" << std::endl;
        std::cout << "  Min time: " << min_time << " ms" << std::endl;
        std::cout << "  Max time: " << max_time << " ms" << std::endl;
        std::cout << "  Avg time: " << mean << " ms" << std::endl;
        std::cout << "  Stddev  : " << stddev << " ms" << std::endl;
    }
}

// 测量fast bootstrap初始化时间
void benchmark_fast_bootstrap(int rank, int world_size, int iterations = 10) {
    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        // 同步所有进程
        MPI_Barrier(MPI_COMM_WORLD);

        // 开始计时
        auto start = std::chrono::high_resolution_clock::now();

        // 创建bootstrap
        auto bootstrap = std::make_shared<fast_bootstrap::FastBootstrap>(rank, world_size);
        fast_bootstrap::UniqueId id;

        // 只在rank 0上创建UniqueId
        if (rank == 0) {
            id = bootstrap->create_unique_id();
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
            std::cout << "Fast Bootstrap Iteration " << i << ": " << max_time << " ms" << std::endl;
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

            std::cout << "\nFast Bootstrap Performance Summary (" << world_size << " processes, excluding first run):" << std::endl;
            std::cout << "  Min time: " << min_time << " ms" << std::endl;
            std::cout << "  Max time: " << max_time << " ms" << std::endl;
            std::cout << "  Avg time: " << mean_without_first << " ms" << std::endl;
            std::cout << "  Stddev  : " << stddev << " ms" << std::endl;
        }

        std::cout << "\nFast Bootstrap Performance Summary (" << world_size << " processes, all runs):" << std::endl;
        std::cout << "  Min time: " << min_time << " ms" << std::endl;
        std::cout << "  Max time: " << max_time << " ms" << std::endl;
        std::cout << "  Avg time: " << mean << " ms" << std::endl;
        std::cout << "  Stddev  : " << stddev << " ms" << std::endl;
    }
}

// 生成性能对比图表
void generate_performance_chart(const std::string& filename, const std::vector<double>& mscclpp_times, const std::vector<double>& fast_times) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // 计算统计信息
    double mscclpp_avg = std::accumulate(mscclpp_times.begin() + 1, mscclpp_times.end(), 0.0) / (mscclpp_times.size() - 1);
    double fast_avg = std::accumulate(fast_times.begin() + 1, fast_times.end(), 0.0) / (fast_times.size() - 1);
    double speedup = mscclpp_avg / fast_avg;

    // 生成HTML图表
    file << "<!DOCTYPE html>\n";
    file << "<html>\n";
    file << "<head>\n";
    file << "    <title>Bootstrap Performance Comparison</title>\n";
    file << "    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n";
    file << "    <style>\n";
    file << "        body { font-family: Arial, sans-serif; margin: 20px; }\n";
    file << "        .chart-container { width: 800px; height: 400px; margin: 20px auto; }\n";
    file << "        .summary { width: 800px; margin: 20px auto; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }\n";
    file << "    </style>\n";
    file << "</head>\n";
    file << "<body>\n";
    file << "    <h1 style=\"text-align: center;\">Bootstrap Performance Comparison</h1>\n";
    file << "    <div class=\"summary\">\n";
    file << "        <h2>Performance Summary</h2>\n";
    file << "        <p><strong>MSCCLPP Average Time:</strong> " << std::fixed << std::setprecision(2) << mscclpp_avg << " ms</p>\n";
    file << "        <p><strong>Fast Bootstrap Average Time:</strong> " << std::fixed << std::setprecision(2) << fast_avg << " ms</p>\n";
    file << "        <p><strong>Speedup:</strong> " << std::fixed << std::setprecision(2) << speedup << "x</p>\n";
    file << "    </div>\n";
    file << "    <div class=\"chart-container\">\n";
    file << "        <canvas id=\"timeChart\"></canvas>\n";
    file << "    </div>\n";
    file << "    <div class=\"chart-container\">\n";
    file << "        <canvas id=\"comparisonChart\"></canvas>\n";
    file << "    </div>\n";
    file << "    <script>\n";
    file << "        // Time series chart\n";
    file << "        const timeCtx = document.getElementById('timeChart').getContext('2d');\n";
    file << "        const timeChart = new Chart(timeCtx, {\n";
    file << "            type: 'line',\n";
    file << "            data: {\n";
    file << "                labels: [";
    for (size_t i = 0; i < mscclpp_times.size(); i++) {
        file << i;
        if (i < mscclpp_times.size() - 1) file << ", ";
    }
    file << "],\n";
    file << "                datasets: [{\n";
    file << "                    label: 'MSCCLPP Bootstrap',\n";
    file << "                    data: [";
    for (size_t i = 0; i < mscclpp_times.size(); i++) {
        file << mscclpp_times[i];
        if (i < mscclpp_times.size() - 1) file << ", ";
    }
    file << "],\n";
    file << "                    borderColor: 'rgb(75, 192, 192)',\n";
    file << "                    tension: 0.1\n";
    file << "                }, {\n";
    file << "                    label: 'Fast Bootstrap',\n";
    file << "                    data: [";
    for (size_t i = 0; i < fast_times.size(); i++) {
        file << fast_times[i];
        if (i < fast_times.size() - 1) file << ", ";
    }
    file << "],\n";
    file << "                    borderColor: 'rgb(255, 99, 132)',\n";
    file << "                    tension: 0.1\n";
    file << "                }]\n";
    file << "            },\n";
    file << "            options: {\n";
    file << "                scales: {\n";
    file << "                    y: {\n";
    file << "                        title: {\n";
    file << "                            display: true,\n";
    file << "                            text: 'Time (ms)'\n";
    file << "                        }\n";
    file << "                    },\n";
    file << "                    x: {\n";
    file << "                        title: {\n";
    file << "                            display: true,\n";
    file << "                            text: 'Iteration'\n";
    file << "                        }\n";
    file << "                    }\n";
    file << "                },\n";
    file << "                plugins: {\n";
    file << "                    title: {\n";
    file << "                        display: true,\n";
    file << "                        text: 'Bootstrap Time per Iteration'\n";
    file << "                    }\n";
    file << "                }\n";
    file << "            }\n";
    file << "        });\n";
    file << "\n";
    file << "        // Comparison chart\n";
    file << "        const compCtx = document.getElementById('comparisonChart').getContext('2d');\n";
    file << "        const compChart = new Chart(compCtx, {\n";
    file << "            type: 'bar',\n";
    file << "            data: {\n";
    file << "                labels: ['Average Bootstrap Time (excluding first run)'],\n";
    file << "                datasets: [{\n";
    file << "                    label: 'MSCCLPP Bootstrap',\n";
    file << "                    data: [" << mscclpp_avg << "],\n";
    file << "                    backgroundColor: 'rgba(75, 192, 192, 0.2)',\n";
    file << "                    borderColor: 'rgb(75, 192, 192)',\n";
    file << "                    borderWidth: 1\n";
    file << "                }, {\n";
    file << "                    label: 'Fast Bootstrap',\n";
    file << "                    data: [" << fast_avg << "],\n";
    file << "                    backgroundColor: 'rgba(255, 99, 132, 0.2)',\n";
    file << "                    borderColor: 'rgb(255, 99, 132)',\n";
    file << "                    borderWidth: 1\n";
    file << "                }]\n";
    file << "            },\n";
    file << "            options: {\n";
    file << "                scales: {\n";
    file << "                    y: {\n";
    file << "                        beginAtZero: true,\n";
    file << "                        title: {\n";
    file << "                            display: true,\n";
    file << "                            text: 'Time (ms)'\n";
    file << "                        }\n";
    file << "                    }\n";
    file << "                },\n";
    file << "                plugins: {\n";
    file << "                    title: {\n";
    file << "                        display: true,\n";
    file << "                        text: 'Average Bootstrap Time Comparison'\n";
    file << "                    }\n";
    file << "                }\n";
    file << "            }\n";
    file << "        });\n";
    file << "    </script>\n";
    file << "</body>\n";
    file << "</html>\n";

    file.close();
    std::cout << "Performance chart generated: " << filename << std::endl;
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
        std::cout << "\n=== MSCCLPP Bootstrap Benchmark ===" << std::endl;
    }

    // 运行mscclpp benchmark
    std::vector<double> mscclpp_times;
    if (rank == 0) {
        mscclpp_times.reserve(iterations);
    }
    benchmark_mscclpp_bootstrap(rank, world_size, iterations);

    // 等待一段时间，确保所有资源都被释放
    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    if (rank == 0) {
        std::cout << "\n=== Fast Bootstrap Benchmark ===" << std::endl;
    }

    // 运行fast bootstrap benchmark
    std::vector<double> fast_times;
    if (rank == 0) {
        fast_times.reserve(iterations);
    }
    benchmark_fast_bootstrap(rank, world_size, iterations);

    // 生成性能对比图表
    if (rank == 0) {
        // Skip chart generation for now to avoid segfault
        std::cout << "\nPerformance Summary:" << std::endl;
        std::cout << "  MSCCLPP Bootstrap avg time: " << std::accumulate(mscclpp_times.begin() + 1, mscclpp_times.end(), 0.0) / (mscclpp_times.size() - 1) << " ms" << std::endl;
        std::cout << "  Fast Bootstrap avg time: " << std::accumulate(fast_times.begin() + 1, fast_times.end(), 0.0) / (fast_times.size() - 1) << " ms" << std::endl;
        std::cout << "  Speedup: " << (std::accumulate(mscclpp_times.begin() + 1, mscclpp_times.end(), 0.0) / (mscclpp_times.size() - 1)) /
                                     (std::accumulate(fast_times.begin() + 1, fast_times.end(), 0.0) / (fast_times.size() - 1)) << "x" << std::endl;
    }

    // 结束MPI
    MPI_Finalize();
    return 0;
}
