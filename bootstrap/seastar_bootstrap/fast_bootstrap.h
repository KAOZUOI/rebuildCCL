#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace fast_bootstrap {

// Unique identifier for bootstrap sessions
struct UniqueId {
    std::array<uint8_t, 16> data;

    UniqueId() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        for (auto& byte : data) {
            byte = dist(gen);
        }
    }

    UniqueId(const std::array<uint8_t, 16>& d) : data(d) {}

    bool operator==(const UniqueId& other) const {
        return data == other.data;
    }
};

// Forward declarations
class Connection;
class FastBootstrap;

// Event loop for asynchronous I/O
class EventLoop {
public:
    EventLoop();
    ~EventLoop();

    void run();
    void stop();
    bool add_fd(int fd, uint32_t events, void* data);
    bool modify_fd(int fd, uint32_t events, void* data);
    bool remove_fd(int fd);

private:
    int epoll_fd_;
    std::atomic<bool> running_;
    std::thread thread_;
};

// Forward declaration of EventLoop
class EventLoop;

// Connection class for TCP communication
class Connection : public std::enable_shared_from_this<Connection> {
public:
    using MessageCallback = std::function<void(const std::vector<uint8_t>&)>;

    Connection(EventLoop& event_loop, int fd);
    ~Connection();

    bool send(const std::vector<uint8_t>& data);
    void set_message_callback(MessageCallback callback);
    void start();
    void close();
    int fd() const { return fd_; }

    // These need to be public so EventLoop can call them
    void handle_read();
    void handle_write();

private:

    EventLoop& event_loop_;
    int fd_;
    MessageCallback message_callback_;
    std::vector<uint8_t> read_buffer_;
    std::vector<uint8_t> write_buffer_;
    std::mutex write_mutex_;
    bool is_writing_;
};

// Fast bootstrap implementation
class FastBootstrap {
public:
    FastBootstrap(int rank, int world_size);
    ~FastBootstrap();

    UniqueId create_unique_id();
    void initialize(const UniqueId& id);
    void barrier();
    void all_gather(const std::vector<uint8_t>& send_data, std::vector<std::vector<uint8_t>>& recv_data);
    void all_to_all(const std::vector<std::vector<uint8_t>>& send_data, std::vector<std::vector<uint8_t>>& recv_data);

private:
    void start_server();
    void connect_to_peers(const UniqueId& id);
    void handle_message(int peer_rank, const std::vector<uint8_t>& message);
    void send_to_peer(int peer_rank, const std::vector<uint8_t>& message);
    void broadcast_unique_id(const UniqueId& id);

    int rank_;
    int world_size_;
    EventLoop event_loop_;
    int server_fd_;
    std::vector<std::shared_ptr<Connection>> connections_;
    std::mutex connections_mutex_;
    std::condition_variable connections_cv_;
    std::atomic<int> connected_peers_;
    std::unordered_map<int, std::vector<std::vector<uint8_t>>> received_messages_;
    std::mutex received_messages_mutex_;
    std::condition_variable received_messages_cv_;
    std::atomic<int> barrier_count_;
    std::mutex barrier_mutex_;
    std::condition_variable barrier_cv_;
};

} // namespace fast_bootstrap
