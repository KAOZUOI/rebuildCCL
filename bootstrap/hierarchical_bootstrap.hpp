#pragma once

#include "seastar_bootstrap.hpp"
#include <seastar/core/gate.hh>
#include <seastar/core/sharded.hh>
#include <seastar/net/tcp.hh>
#include <seastar/core/future-util.hh>
#include <seastar/core/metrics.hh>
#include <unordered_map>
#include <queue>

namespace bootstrap {

// 分层次的Bootstrap服务实现
class HierarchicalBootstrap : public BootstrapService {
public:
    HierarchicalBootstrap() : _stopped(false) {}
    virtual ~HierarchicalBootstrap() = default;

    // 初始化服务
    virtual seastar::future<> init(int rank, int total_ranks, const seastar::sstring& address, int port) override {
        _rank = rank;
        _total_ranks = total_ranks;
        _address = address;
        _port = port;
        _my_info = NodeInfo{_rank, _address, _port};

        // 计算树形结构
        compute_tree_topology();

        // 初始化节点信息数组
        _nodes.resize(_total_ranks);
        _nodes[_rank] = _my_info;

        // 初始化消息队列
        _message_queues.resize(_total_ranks);

        return seastar::make_ready_future<>();
    }

    // 启动bootstrap过程
    virtual seastar::future<> start() override {
        // 启动监听服务
        return start_server().then([this] {
            // 等待所有预期的连接建立
            return seastar::sleep(std::chrono::milliseconds(100 * _rank)).then([this] {
                return connect_to_parent().then([this] {
                    // 执行分层次的bootstrap过程
                    return bootstrap_process();
                });
            });
        });
    }

    // 停止服务
    virtual seastar::future<> stop() override {
        _stopped = true;
        return _connection_gate.close().then([this] {
            return seastar::parallel_for_each(_connections, [](auto& conn_pair) {
                return conn_pair.second->shutdown_input().then([&conn_pair] {
                    return conn_pair.second->shutdown_output();
                });
            }).then([this] {
                _connections.clear();
                if (_server_socket) {
                    return _server_socket.shutdown();
                }
                return seastar::make_ready_future<>();
            });
        });
    }

    // 获取节点信息
    virtual const std::vector<NodeInfo>& get_nodes() const override {
        return _nodes;
    }

    // 发送消息到指定节点
    virtual seastar::future<> send(int target_rank, const seastar::sstring& message) override {
        if (_connections.find(target_rank) == _connections.end()) {
            // 如果没有直接连接，通过树形结构路由
            return route_message(target_rank, message);
        }

        // 构造消息格式：[消息长度][消息内容]
        seastar::temporary_buffer<char> buf(sizeof(uint32_t) + message.size());
        uint32_t msg_len = message.size();
        std::memcpy(buf.get_write(), &msg_len, sizeof(uint32_t));
        std::memcpy(buf.get_write() + sizeof(uint32_t), message.data(), message.size());

        return _connections[target_rank]->output().write(std::move(buf));
    }

    // 接收来自指定节点的消息
    virtual seastar::future<seastar::sstring> recv(int source_rank) override {
        // 检查消息队列中是否已有消息
        if (!_message_queues[source_rank].empty()) {
            auto msg = std::move(_message_queues[source_rank].front());
            _message_queues[source_rank].pop();
            return seastar::make_ready_future<seastar::sstring>(std::move(msg));
        }

        // 如果没有消息，创建一个promise并等待
        auto [promise, future] = seastar::promise<seastar::sstring>().get_future();
        _pending_receives[source_rank].push(std::move(promise));
        return future;
    }

    // 执行集体通信操作：广播
    virtual seastar::future<seastar::sstring> broadcast(int root, const seastar::sstring& message) override {
        if (_rank == root) {
            // 如果是根节点，向所有子节点发送消息
            std::vector<seastar::future<>> futures;
            for (int child : _children) {
                futures.push_back(send(child, message));
            }
            return seastar::when_all_succeed(futures.begin(), futures.end())
                .then([message] { return message; });
        } else {
            // 如果不是根节点，从父节点接收消息，然后转发给子节点
            return recv(_parent).then([this, message = message](const seastar::sstring& received) {
                std::vector<seastar::future<>> futures;
                for (int child : _children) {
                    futures.push_back(send(child, received));
                }
                return seastar::when_all_succeed(futures.begin(), futures.end())
                    .then([received] { return received; });
            });
        }
    }

    // 执行集体通信操作：allgather
    virtual seastar::future<std::vector<seastar::sstring>> allgather(const seastar::sstring& local_data) override {
        std::vector<seastar::sstring> result(_total_ranks);
        result[_rank] = local_data;

        // 首先向上收集数据
        return gather_up(local_data).then([this, result = std::move(result)](std::vector<seastar::sstring> gathered) mutable {
            // 合并收集到的数据
            for (size_t i = 0; i < gathered.size(); ++i) {
                if (!gathered[i].empty()) {
                    result[i] = std::move(gathered[i]);
                }
            }

            // 然后向下广播完整数据
            return broadcast_down(result).then([result = std::move(result)] {
                return result;
            });
        });
    }

    // 执行集体通信操作：barrier
    virtual seastar::future<> barrier() override {
        // 实现一个简单的树形barrier
        if (_children.empty()) {
            // 叶子节点直接向父节点发送就绪信号
            if (_rank != 0) {
                return send(_parent, "BARRIER_READY").then([this] {
                    return recv(_parent);
                }).then([](const seastar::sstring&) {
                    return seastar::make_ready_future<>();
                });
            }
            return seastar::make_ready_future<>();
        } else {
            // 非叶子节点等待所有子节点就绪
            std::vector<seastar::future<seastar::sstring>> futures;
            for (int child : _children) {
                futures.push_back(recv(child));
            }

            return seastar::when_all_succeed(futures.begin(), futures.end())
                .then([this](std::vector<seastar::sstring>) {
                    // 向父节点发送就绪信号
                    if (_rank != 0) {
                        return send(_parent, "BARRIER_READY").then([this] {
                            return recv(_parent);
                        });
                    }
                    return seastar::make_ready_future<seastar::sstring>("");
                }).then([this](const seastar::sstring&) {
                    // 向所有子节点发送完成信号
                    std::vector<seastar::future<>> send_futures;
                    for (int child : _children) {
                        send_futures.push_back(send(child, "BARRIER_DONE"));
                    }
                    return seastar::when_all_succeed(send_futures.begin(), send_futures.end());
                });
        }
    }

private:
    int _rank;
    int _total_ranks;
    seastar::sstring _address;
    int _port;
    NodeInfo _my_info;
    std::vector<NodeInfo> _nodes;
    bool _stopped;

    // 树形拓扑结构
    int _parent = -1;
    std::vector<int> _children;
    int _tree_degree = 4;  // 每个节点的最大子节点数

    // 网络相关
    seastar::server_socket _server_socket;
    std::unordered_map<int, seastar::connected_socket> _connections;
    seastar::gate _connection_gate;

    // 消息队列
    std::vector<std::queue<seastar::sstring>> _message_queues;
    std::unordered_map<int, std::queue<seastar::promise<seastar::sstring>>> _pending_receives;

    // 计算树形拓扑结构
    void compute_tree_topology() {
        if (_rank > 0) {
            _parent = (_rank - 1) / _tree_degree;
        }

        for (int i = 1; i <= _tree_degree; ++i) {
            int child_rank = _rank * _tree_degree + i;
            if (child_rank < _total_ranks) {
                _children.push_back(child_rank);
            }
        }
    }

    // 启动服务器监听
    seastar::future<> start_server() {
        return seastar::do_with(seastar::listen(seastar::make_ipv4_address({_port})),
            [this](auto& listener) {
                _server_socket = std::move(listener);
                return seastar::keep_doing([this] {
                    return _server_socket.accept().then([this](seastar::accept_result ar) {
                        // 处理新连接
                        auto conn = std::move(ar.connection);
                        auto addr = ar.remote_address;

                        return handle_connection(std::move(conn));
                    });
                }).handle_exception([this](std::exception_ptr ep) {
                    if (!_stopped) {
                        seastar::log.error("Error in server: {}", ep);
                    }
                });
            });
    }

    // 处理新连接
    seastar::future<> handle_connection(seastar::connected_socket conn) {
        return seastar::do_with(std::move(conn), [this](auto& conn) {
            return seastar::do_with(conn.input(), [this, &conn](auto& input) {
                return seastar::keep_doing([this, &input, &conn] {
                    // 读取消息长度
                    return input.read_exactly(sizeof(uint32_t)).then([this, &input, &conn](seastar::temporary_buffer<char> size_buf) {
                        if (size_buf.empty()) {
                            return seastar::make_ready_future<seastar::stop_iteration>(seastar::stop_iteration::yes);
                        }

                        uint32_t msg_size;
                        std::memcpy(&msg_size, size_buf.get(), sizeof(uint32_t));

                        // 读取消息内容
                        return input.read_exactly(msg_size).then([this, &conn](seastar::temporary_buffer<char> msg_buf) {
                            if (msg_buf.empty()) {
                                return seastar::make_ready_future<seastar::stop_iteration>(seastar::stop_iteration::yes);
                            }

                            seastar::sstring msg(msg_buf.get(), msg_buf.size());

                            // 处理消息
                            // TODO: 解析消息头，确定源rank和目标rank

                            return seastar::make_ready_future<seastar::stop_iteration>(seastar::stop_iteration::no);
                        });
                    });
                });
            });
        });
    }

    // 连接到父节点
    seastar::future<> connect_to_parent() {
        if (_parent < 0 || _parent == _rank) {
            return seastar::make_ready_future<>();
        }

        // 假设父节点的地址和端口是已知的
        // 在实际应用中，这可能需要通过配置文件或服务发现机制获取
        seastar::sstring parent_addr = _address; // 在这个demo中使用相同地址
        int parent_port = _port - (_rank - _parent); // 简单计算父节点端口

        seastar::log.info("Rank {} connecting to parent rank {} at {}:{}",
                         _rank, _parent, parent_addr, parent_port);

        return seastar::connect(seastar::make_ipv4_address({parent_addr, (uint16_t)parent_port}))
            .then([this](seastar::connected_socket socket) {
                _connections[_parent] = std::move(socket);

                // 发送自己的身份信息给父节点
                seastar::sstring identity_msg = seastar::format("IDENTITY:{}", _rank);
                return send(_parent, identity_msg);
            });
    }

    // 执行bootstrap过程
    seastar::future<> bootstrap_process() {
        // 1. 收集本地节点信息
        // 2. 通过树形结构传播节点信息
        // 3. 建立必要的直接连接

        seastar::log.info("Rank {} starting bootstrap process", _rank);

        // 如果是根节点（rank 0），启动bootstrap过程
        if (_rank == 0) {
            // 根节点已经知道自己的信息
            _nodes[0] = _my_info;

            // 等待所有子节点连接并发送身份信息
            std::vector<seastar::future<seastar::sstring>> identity_futures;
            for (int child : _children) {
                identity_futures.push_back(recv(child));
            }

            return seastar::when_all_succeed(identity_futures.begin(), identity_futures.end())
                .then([this](std::vector<seastar::sstring> identities) {
                    // 处理子节点的身份信息
                    for (const auto& identity : identities) {
                        if (identity.substr(0, 9) == "IDENTITY:") {
                            int child_rank = std::stoi(identity.substr(9));
                            // 在实际应用中，这里会解析更多信息
                            seastar::log.info("Root received identity from rank {}", child_rank);
                        }
                    }

                    // 收集所有节点信息
                    return collect_all_node_info();
                })
                .then([this] {
                    // 广播完整的节点信息给所有子节点
                    return broadcast_node_info();
                });
        } else {
            // 非根节点等待从父节点接收完整的节点信息
            return recv(_parent).then([this](seastar::sstring node_info) {
                // 处理节点信息
                if (node_info.substr(0, 10) == "NODE_INFO:") {
                    // 解析节点信息
                    // 在实际应用中，这里会解析更详细的信息
                    seastar::log.info("Rank {} received node info from parent", _rank);

                    // 转发给子节点
                    std::vector<seastar::future<>> forwards;
                    for (int child : _children) {
                        forwards.push_back(send(child, node_info));
                    }
                    return seastar::when_all_succeed(forwards.begin(), forwards.end());
                }
                return seastar::make_ready_future<>();
            });
        }
    }

    // 收集所有节点信息
    seastar::future<> collect_all_node_info() {
        // 在实际应用中，这里会收集更详细的节点信息
        seastar::log.info("Rank {} collecting all node info", _rank);

        // 模拟收集节点信息
        for (int i = 0; i < _total_ranks; ++i) {
            if (i != _rank) {
                _nodes[i] = NodeInfo{i, _address, _port + (i - _rank)};
            }
        }

        return seastar::make_ready_future<>();
    }

    // 广播节点信息
    seastar::future<> broadcast_node_info() {
        // 在实际应用中，这里会序列化和广播完整的节点信息
        seastar::log.info("Rank {} broadcasting node info", _rank);

        // 简单模拟：发送一个标记消息
        std::vector<seastar::future<>> broadcasts;
        for (int child : _children) {
            broadcasts.push_back(send(child, "NODE_INFO:COMPLETE"));
        }

        return seastar::when_all_succeed(broadcasts.begin(), broadcasts.end());
    }

    // 通过树形结构路由消息
    seastar::future<> route_message(int target_rank, const seastar::sstring& message) {
        // 确定消息应该路由到哪个节点
        int next_hop;

        if (is_ancestor(target_rank)) {
            // 如果目标是祖先节点，发送给父节点
            next_hop = _parent;
        } else if (is_descendant(target_rank)) {
            // 如果目标是后代节点，找到正确的子节点路径
            next_hop = find_child_path(target_rank);
        } else {
            // 否则，发送给父节点
            next_hop = _parent;
        }

        // 构造路由消息
        seastar::sstring routed_msg = seastar::format("ROUTE:{}:{}", target_rank, message);
        return send(next_hop, routed_msg);
    }

    // 检查一个节点是否是当前节点的祖先
    bool is_ancestor(int rank) {
        int current = _parent;
        while (current >= 0) {
            if (current == rank) return true;
            current = (current - 1) / _tree_degree;
        }
        return false;
    }

    // 检查一个节点是否是当前节点的后代
    bool is_descendant(int rank) {
        // 检查rank是否在以当前节点为根的子树中
        int min_rank = _rank * _tree_degree + 1;
        int max_rank = min_rank + _tree_degree - 1;

        if (rank >= min_rank && rank <= max_rank && rank < _total_ranks) {
            return true;
        }

        // 递归检查子节点
        for (int child : _children) {
            min_rank = child * _tree_degree + 1;
            max_rank = min_rank + _tree_degree - 1;
            if (rank >= min_rank && rank <= max_rank && rank < _total_ranks) {
                return true;
            }
        }

        return false;
    }

    // 找到通向目标节点的子节点路径
    int find_child_path(int target_rank) {
        for (int child : _children) {
            if (target_rank == child) return child;

            int min_rank = child * _tree_degree + 1;
            int max_rank = min_rank + _tree_degree - 1;
            if (target_rank >= min_rank && target_rank <= max_rank && target_rank < _total_ranks) {
                return child;
            }
        }

        return _parent;  // 默认返回父节点
    }

    // 向上收集数据（用于allgather）
    seastar::future<std::vector<seastar::sstring>> gather_up(const seastar::sstring& local_data) {
        std::vector<seastar::sstring> result(_total_ranks);
        result[_rank] = local_data;

        // 如果是叶子节点，直接将数据发送给父节点
        if (_children.empty()) {
            if (_rank != 0) {
                // 构造消息：GATHER:rank:data
                seastar::sstring msg = seastar::format("GATHER:{}:{}", _rank, local_data);
                return send(_parent, msg).then([result = std::move(result)] {
                    return result;
                });
            }
            return seastar::make_ready_future<std::vector<seastar::sstring>>(std::move(result));
        }

        // 非叶子节点，等待所有子节点的数据
        std::vector<seastar::future<seastar::sstring>> child_futures;
        for (int child : _children) {
            child_futures.push_back(recv(child));
        }

        return seastar::when_all_succeed(child_futures.begin(), child_futures.end())
            .then([this, result = std::move(result), local_data](std::vector<seastar::sstring> child_data) mutable {
                // 处理子节点数据
                for (const auto& data : child_data) {
                    if (data.substr(0, 7) == "GATHER:") {
                        size_t pos = data.find(':', 7);
                        if (pos != seastar::sstring::npos) {
                            int src_rank = std::stoi(data.substr(7, pos - 7));
                            seastar::sstring value = data.substr(pos + 1);
                            result[src_rank] = value;
                        }
                    }
                }

                // 如果不是根节点，将收集到的数据发送给父节点
                if (_rank != 0) {
                    // 构造包含所有收集到的数据的消息
                    seastar::sstring gathered_data;
                    for (int i = 0; i < _total_ranks; ++i) {
                        if (!result[i].empty()) {
                            gathered_data += seastar::format("GATHER:{}:{};", i, result[i]);
                        }
                    }

                    return send(_parent, gathered_data).then([result = std::move(result)] {
                        return result;
                    });
                }

                return seastar::make_ready_future<std::vector<seastar::sstring>>(std::move(result));
            });
    }

    // 向下广播数据（用于allgather）
    seastar::future<> broadcast_down(const std::vector<seastar::sstring>& data) {
        // 如果是叶子节点，不需要广播
        if (_children.empty()) {
            return seastar::make_ready_future<>();
        }

        // 构造广播消息
        seastar::sstring broadcast_msg = "BROADCAST:";
        for (size_t i = 0; i < data.size(); ++i) {
            if (!data[i].empty()) {
                broadcast_msg += seastar::format("{}:{};", i, data[i]);
            }
        }

        // 向所有子节点广播
        std::vector<seastar::future<>> broadcast_futures;
        for (int child : _children) {
            broadcast_futures.push_back(send(child, broadcast_msg));
        }

        return seastar::when_all_succeed(broadcast_futures.begin(), broadcast_futures.end());
    }
};

} // namespace bootstrap
