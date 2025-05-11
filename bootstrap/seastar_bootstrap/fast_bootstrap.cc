#include "fast_bootstrap.h"

namespace fast_bootstrap {

// EventLoop implementation
EventLoop::EventLoop() : running_(false) {
    epoll_fd_ = epoll_create1(0);
    if (epoll_fd_ == -1) {
        throw std::runtime_error("Failed to create epoll file descriptor");
    }
}

EventLoop::~EventLoop() {
    stop();
    close(epoll_fd_);
}

void EventLoop::run() {
    running_ = true;
    thread_ = std::thread([this]() {
        const int MAX_EVENTS = 64;
        struct epoll_event events[MAX_EVENTS];

        while (running_) {
            int n = epoll_wait(epoll_fd_, events, MAX_EVENTS, 100); // 100ms timeout
            for (int i = 0; i < n; i++) {
                auto* conn = static_cast<Connection*>(events[i].data.ptr);
                if (events[i].events & EPOLLIN) {
                    conn->handle_read();
                }
                if (events[i].events & EPOLLOUT) {
                    conn->handle_write();
                }
                if (events[i].events & (EPOLLERR | EPOLLHUP)) {
                    conn->close();
                }
            }
        }
    });
}

void EventLoop::stop() {
    if (running_) {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }
}

bool EventLoop::add_fd(int fd, uint32_t events, void* data) {
    struct epoll_event ev;
    ev.events = events;
    ev.data.ptr = data;
    return epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) == 0;
}

bool EventLoop::modify_fd(int fd, uint32_t events, void* data) {
    struct epoll_event ev;
    ev.events = events;
    ev.data.ptr = data;
    return epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &ev) == 0;
}

bool EventLoop::remove_fd(int fd) {
    return epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr) == 0;
}

// Connection implementation
Connection::Connection(EventLoop& event_loop, int fd)
    : event_loop_(event_loop), fd_(fd), is_writing_(false) {
    // Set socket to non-blocking mode
    int flags = fcntl(fd_, F_GETFL, 0);
    fcntl(fd_, F_SETFL, flags | O_NONBLOCK);

    // Disable Nagle's algorithm
    int flag = 1;
    setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(int));

    // Set TCP_QUICKACK
    setsockopt(fd_, IPPROTO_TCP, TCP_QUICKACK, &flag, sizeof(int));

    // Set keep-alive
    setsockopt(fd_, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(int));

    // Allocate read buffer
    read_buffer_.resize(4096);
}

Connection::~Connection() {
    close();
}

void Connection::start() {
    event_loop_.add_fd(fd_, EPOLLIN, this);
}

void Connection::close() {
    if (fd_ >= 0) {
        event_loop_.remove_fd(fd_);
        ::close(fd_);
        fd_ = -1;
    }
}

bool Connection::send(const std::vector<uint8_t>& data) {
    if (fd_ < 0) return false;

    std::lock_guard<std::mutex> lock(write_mutex_);
    bool was_empty = write_buffer_.empty();
    write_buffer_.insert(write_buffer_.end(), data.begin(), data.end());

    if (was_empty && !is_writing_) {
        is_writing_ = true;
        event_loop_.modify_fd(fd_, EPOLLIN | EPOLLOUT, this);
    }

    return true;
}

void Connection::set_message_callback(MessageCallback callback) {
    message_callback_ = std::move(callback);
}

void Connection::handle_read() {
    if (fd_ < 0) return;

    ssize_t n = read(fd_, read_buffer_.data(), read_buffer_.size());
    if (n > 0) {
        if (message_callback_) {
            message_callback_(std::vector<uint8_t>(read_buffer_.data(), read_buffer_.data() + n));
        }
    } else if (n == 0) {
        // Connection closed by peer
        close();
    }
}

void Connection::handle_write() {
    if (fd_ < 0) return;

    std::lock_guard<std::mutex> lock(write_mutex_);
    if (!write_buffer_.empty()) {
        ssize_t n = write(fd_, write_buffer_.data(), write_buffer_.size());
        if (n > 0) {
            write_buffer_.erase(write_buffer_.begin(), write_buffer_.begin() + n);
        }
    }

    if (write_buffer_.empty()) {
        is_writing_ = false;
        event_loop_.modify_fd(fd_, EPOLLIN, this);
    }
}

// FastBootstrap implementation
FastBootstrap::FastBootstrap(int rank, int world_size)
    : rank_(rank), world_size_(world_size), server_fd_(-1), connected_peers_(0), barrier_count_(0) {
    connections_.resize(world_size_);
    event_loop_.run();
}

FastBootstrap::~FastBootstrap() {
    if (server_fd_ >= 0) {
        close(server_fd_);
    }
    event_loop_.stop();
}

UniqueId FastBootstrap::create_unique_id() {
    return UniqueId();
}

void FastBootstrap::start_server() {
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        throw std::runtime_error("Failed to create server socket");
    }

    // Allow reuse of address
    int opt = 1;
    if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        close(server_fd_);
        throw std::runtime_error("Failed to set SO_REUSEADDR");
    }

    // Set non-blocking mode
    int flags = fcntl(server_fd_, F_GETFL, 0);
    fcntl(server_fd_, F_SETFL, flags | O_NONBLOCK);

    // Bind to port
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(12345 + rank_); // Use different port for each rank

    if (bind(server_fd_, (struct sockaddr*)&address, sizeof(address)) < 0) {
        close(server_fd_);
        throw std::runtime_error("Failed to bind server socket");
    }

    if (listen(server_fd_, world_size_) < 0) {
        close(server_fd_);
        throw std::runtime_error("Failed to listen on server socket");
    }

    // Accept connections in a separate thread
    std::thread accept_thread([this]() {
        while (connected_peers_ < world_size_ - 1) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);

            if (client_fd < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                break;
            }

            // Receive peer rank
            uint32_t peer_rank;
            if (recv(client_fd, &peer_rank, sizeof(peer_rank), 0) != sizeof(peer_rank)) {
                close(client_fd);
                continue;
            }

            // Create connection
            auto conn = std::make_shared<Connection>(event_loop_, client_fd);
            conn->set_message_callback([this, peer_rank](const std::vector<uint8_t>& message) {
                handle_message(peer_rank, message);
            });
            conn->start();

            // Store connection
            std::lock_guard<std::mutex> lock(connections_mutex_);
            connections_[peer_rank] = conn;
            connected_peers_++;
            connections_cv_.notify_all();
        }
    });
    accept_thread.detach();
}

void FastBootstrap::connect_to_peers(const UniqueId& /*id*/) {
    start_server();

    // Connect to peers with lower ranks
    for (int peer_rank = 0; peer_rank < rank_; peer_rank++) {
        int peer_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (peer_fd < 0) {
            throw std::runtime_error("Failed to create socket for peer connection");
        }

        struct sockaddr_in peer_addr;
        peer_addr.sin_family = AF_INET;
        peer_addr.sin_addr.s_addr = inet_addr("127.0.0.1"); // Assuming local connections
        peer_addr.sin_port = htons(12345 + peer_rank);

        // Connect with retry
        int retries = 0;
        while (retries < 10) {
            if (connect(peer_fd, (struct sockaddr*)&peer_addr, sizeof(peer_addr)) == 0) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            retries++;
        }

        if (retries == 10) {
            close(peer_fd);
            throw std::runtime_error("Failed to connect to peer " + std::to_string(peer_rank));
        }

        // Send our rank
        uint32_t our_rank = rank_;
        if (send(peer_fd, &our_rank, sizeof(our_rank), 0) != sizeof(our_rank)) {
            close(peer_fd);
            throw std::runtime_error("Failed to send rank to peer " + std::to_string(peer_rank));
        }

        // Create connection
        auto conn = std::make_shared<Connection>(event_loop_, peer_fd);
        conn->set_message_callback([this, peer_rank](const std::vector<uint8_t>& message) {
            handle_message(peer_rank, message);
        });
        conn->start();

        // Store connection
        std::lock_guard<std::mutex> lock(connections_mutex_);
        connections_[peer_rank] = conn;
        connected_peers_++;
        connections_cv_.notify_all();
    }

    // Wait for all connections to be established
    std::unique_lock<std::mutex> lock(connections_mutex_);
    connections_cv_.wait(lock, [this]() { return connected_peers_ == world_size_ - 1; });
}

void FastBootstrap::initialize(const UniqueId& id) {
    connect_to_peers(id);
}

void FastBootstrap::handle_message(int peer_rank, const std::vector<uint8_t>& message) {
    // Simple message handling - just store the message
    std::lock_guard<std::mutex> lock(received_messages_mutex_);
    received_messages_[peer_rank].push_back(message);
    received_messages_cv_.notify_all();
}

void FastBootstrap::send_to_peer(int peer_rank, const std::vector<uint8_t>& message) {
    if (peer_rank == rank_) return;

    std::lock_guard<std::mutex> lock(connections_mutex_);
    if (connections_[peer_rank]) {
        connections_[peer_rank]->send(message);
    }
}

void FastBootstrap::barrier() {
    // Simple barrier implementation
    barrier_count_++;

    // Notify all peers
    std::vector<uint8_t> barrier_msg = {0}; // Simple barrier message
    for (int peer_rank = 0; peer_rank < world_size_; peer_rank++) {
        if (peer_rank != rank_) {
            send_to_peer(peer_rank, barrier_msg);
        }
    }

    // Wait for messages from all peers
    std::unique_lock<std::mutex> lock(received_messages_mutex_);
    for (int peer_rank = 0; peer_rank < world_size_; peer_rank++) {
        if (peer_rank != rank_) {
            received_messages_cv_.wait(lock, [this, peer_rank]() {
                return !received_messages_[peer_rank].empty();
            });
            received_messages_[peer_rank].clear();
        }
    }
}

void FastBootstrap::all_gather(const std::vector<uint8_t>& send_data, std::vector<std::vector<uint8_t>>& recv_data) {
    recv_data.resize(world_size_);
    recv_data[rank_] = send_data;

    // Send data to all peers
    for (int peer_rank = 0; peer_rank < world_size_; peer_rank++) {
        if (peer_rank != rank_) {
            send_to_peer(peer_rank, send_data);
        }
    }

    // Receive data from all peers
    std::unique_lock<std::mutex> lock(received_messages_mutex_);
    for (int peer_rank = 0; peer_rank < world_size_; peer_rank++) {
        if (peer_rank != rank_) {
            received_messages_cv_.wait(lock, [this, peer_rank]() {
                return !received_messages_[peer_rank].empty();
            });
            recv_data[peer_rank] = received_messages_[peer_rank].front();
            received_messages_[peer_rank].clear();
        }
    }
}

void FastBootstrap::all_to_all(const std::vector<std::vector<uint8_t>>& send_data, std::vector<std::vector<uint8_t>>& recv_data) {
    recv_data.resize(world_size_);
    recv_data[rank_] = send_data[rank_];

    // Send data to all peers
    for (int peer_rank = 0; peer_rank < world_size_; peer_rank++) {
        if (peer_rank != rank_) {
            send_to_peer(peer_rank, send_data[peer_rank]);
        }
    }

    // Receive data from all peers
    std::unique_lock<std::mutex> lock(received_messages_mutex_);
    for (int peer_rank = 0; peer_rank < world_size_; peer_rank++) {
        if (peer_rank != rank_) {
            received_messages_cv_.wait(lock, [this, peer_rank]() {
                return !received_messages_[peer_rank].empty();
            });
            recv_data[peer_rank] = received_messages_[peer_rank].front();
            received_messages_[peer_rank].clear();
        }
    }
}

} // namespace fast_bootstrap
