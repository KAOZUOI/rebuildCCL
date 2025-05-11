#pragma once

#include <seastar/core/future.hh>
#include <seastar/core/distributed.hh>
#include <seastar/core/reactor.hh>
#include <seastar/core/app_template.hh>
#include <seastar/core/temporary_buffer.hh>
#include <seastar/core/sleep.hh>
#include <seastar/core/shared_ptr.hh>
#include <seastar/net/api.hh>
#include <seastar/core/print.hh>
#include <seastar/core/sstring.hh>
#include <seastar/util/log.hh>

#include <unordered_map>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <functional>

namespace bootstrap {

// 节点信息
struct NodeInfo {
    int rank;                   // 节点的rank
    seastar::sstring address;   // 节点的地址
    int port;                   // 节点的端口
    
    // 序列化为字符串
    seastar::sstring serialize() const {
        return seastar::format("{}:{}:{}", rank, address, port);
    }
    
    // 从字符串反序列化
    static NodeInfo deserialize(const seastar::sstring& data) {
        auto parts = split(data, ':');
        if (parts.size() != 3) {
            throw std::runtime_error("Invalid node info format");
        }
        return {
            std::stoi(parts[0]),
            parts[1],
            std::stoi(parts[2])
        };
    }
    
private:
    // 分割字符串
    static std::vector<seastar::sstring> split(const seastar::sstring& str, char delimiter) {
        std::vector<seastar::sstring> result;
        size_t start = 0;
        size_t end = str.find(delimiter);
        
        while (end != seastar::sstring::npos) {
            result.push_back(str.substr(start, end - start));
            start = end + 1;
            end = str.find(delimiter, start);
        }
        
        result.push_back(str.substr(start));
        return result;
    }
};

// 唯一ID，用于标识一个通信组
struct UniqueId {
    uint64_t id;
    
    // 创建随机唯一ID
    static UniqueId create() {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dis;
        return {dis(gen)};
    }
    
    // 序列化为字符串
    seastar::sstring serialize() const {
        return seastar::to_sstring(id);
    }
    
    // 从字符串反序列化
    static UniqueId deserialize(const seastar::sstring& data) {
        return {std::stoull(data)};
    }
};

// 连接类型
enum class ConnectionType {
    TCP,        // TCP连接
    RDMA,       // RDMA连接
    LOCAL       // 本地连接
};

// 前向声明
class BootstrapService;

// Bootstrap服务接口
class BootstrapService {
public:
    virtual ~BootstrapService() = default;
    
    // 初始化服务
    virtual seastar::future<> init(int rank, int total_ranks, const seastar::sstring& address, int port) = 0;
    
    // 启动bootstrap过程
    virtual seastar::future<> start() = 0;
    
    // 停止服务
    virtual seastar::future<> stop() = 0;
    
    // 获取节点信息
    virtual const std::vector<NodeInfo>& get_nodes() const = 0;
    
    // 发送消息到指定节点
    virtual seastar::future<> send(int target_rank, const seastar::sstring& message) = 0;
    
    // 接收来自指定节点的消息
    virtual seastar::future<seastar::sstring> recv(int source_rank) = 0;
    
    // 执行集体通信操作：广播
    virtual seastar::future<seastar::sstring> broadcast(int root, const seastar::sstring& message) = 0;
    
    // 执行集体通信操作：allgather
    virtual seastar::future<std::vector<seastar::sstring>> allgather(const seastar::sstring& local_data) = 0;
    
    // 执行集体通信操作：barrier
    virtual seastar::future<> barrier() = 0;
};

} // namespace bootstrap
