#include "hierarchical_bootstrap.hpp"
#include <seastar/core/app_template.hh>
#include <seastar/core/reactor.hh>
#include <seastar/core/distributed.hh>
#include <seastar/core/sleep.hh>
#include <seastar/util/log.hh>
#include <iostream>
#include <chrono>

using namespace bootstrap;
using namespace seastar;

// 演示程序
class BootstrapDemo {
public:
    BootstrapDemo() : _bootstrap(std::make_unique<HierarchicalBootstrap>()) {}
    
    // 初始化
    future<> init(int rank, int total_ranks, const sstring& address, int port) {
        _rank = rank;
        _total_ranks = total_ranks;
        
        std::cout << format("Initializing node rank={} at {}:{}\n", rank, address, port);
        
        return _bootstrap->init(rank, total_ranks, address, port);
    }
    
    // 启动bootstrap过程
    future<> start() {
        std::cout << format("Starting bootstrap process on rank {}\n", _rank);
        
        return _bootstrap->start().then([this] {
            std::cout << format("Bootstrap completed on rank {}\n", _rank);
            
            // 执行一些测试操作
            return test_operations();
        });
    }
    
    // 停止服务
    future<> stop() {
        std::cout << format("Stopping bootstrap service on rank {}\n", _rank);
        return _bootstrap->stop();
    }
    
private:
    int _rank;
    int _total_ranks;
    std::unique_ptr<BootstrapService> _bootstrap;
    
    // 测试各种操作
    future<> test_operations() {
        // 等待所有节点完成bootstrap
        return _bootstrap->barrier().then([this] {
            std::cout << format("Barrier completed on rank {}\n", _rank);
            
            // 测试广播
            sstring message;
            if (_rank == 0) {
                message = format("Broadcast message from rank {}", _rank);
            }
            
            return _bootstrap->broadcast(0, message);
        }).then([this](sstring received) {
            std::cout << format("Rank {} received broadcast: {}\n", _rank, received);
            
            // 测试allgather
            sstring local_data = format("Data from rank {}", _rank);
            return _bootstrap->allgather(local_data);
        }).then([this](std::vector<sstring> gathered) {
            std::cout << format("Rank {} received allgather data:\n", _rank);
            for (size_t i = 0; i < gathered.size(); ++i) {
                std::cout << format("  Rank {}: {}\n", i, gathered[i]);
            }
            
            // 测试点对点通信
            if (_rank == 0) {
                std::vector<future<>> sends;
                for (int i = 1; i < _total_ranks; ++i) {
                    sstring msg = format("Message from rank 0 to rank {}", i);
                    sends.push_back(_bootstrap->send(i, msg));
                }
                return when_all_succeed(sends.begin(), sends.end());
            } else {
                return _bootstrap->recv(0).then([this](sstring msg) {
                    std::cout << format("Rank {} received message: {}\n", _rank, msg);
                    return make_ready_future<>();
                });
            }
        });
    }
};

// 主函数
int main(int argc, char** argv) {
    app_template app;
    
    // 添加命令行参数
    app.add_options()
        ("rank", program_options::value<int>()->default_value(0), "Node rank")
        ("total-ranks", program_options::value<int>()->default_value(4), "Total number of ranks")
        ("address", program_options::value<std::string>()->default_value("127.0.0.1"), "Node address")
        ("base-port", program_options::value<int>()->default_value(10000), "Base port number");
    
    return app.run(argc, argv, [&app] {
        auto& config = app.configuration();
        int rank = config["rank"].as<int>();
        int total_ranks = config["total-ranks"].as<int>();
        sstring address = config["address"].as<std::string>();
        int base_port = config["base-port"].as<int>();
        int port = base_port + rank;
        
        return do_with(BootstrapDemo(), [rank, total_ranks, address, port](BootstrapDemo& demo) {
            return demo.init(rank, total_ranks, address, port)
                .then([&demo] {
                    return demo.start();
                })
                .then([&demo] {
                    // 运行一段时间后停止
                    return sleep(std::chrono::seconds(10)).then([&demo] {
                        return demo.stop();
                    });
                });
        });
    });
}
