#include <iostream>
#include <vector>
#include <memory>
#include <ucxx/api.h>

static uint16_t listener_port = 52347;

class Node {
 public:
  int rank;
  int totalRanks;

  Node(int rank, int totalRanks) : rank(rank), totalRanks(totalRanks) {}
};

class ListenerCTX {
 private:
  std::shared_ptr<ucxx::Worker> _worker;
  std::shared_ptr<ucxx::Listener> _listener;
  std::shared_ptr<ucxx::Endpoint> _endpoint;
 public:
  explicit ListenerCTX(std::shared_ptr<ucxx::Worker> worker) : _worker(worker) {}
  ~ListenerCTX() { releaseEndpoint(); }

  void setListener(std::shared_ptr<ucxx::Listener> listener) { _listener = listener; }
  std::shared_ptr<ucxx::Listener> getListener() { return _listener; }
  std::shared_ptr<ucxx::Endpoint> getEndpoint() { return _endpoint; }
  bool isAvailable() const { return _endpoint == nullptr; }

  void createEndpointFromConnRequest(ucp_conn_request_h conn_request) {
    if (!isAvailable()) throw std::runtime_error("Listener context already has an endpoint");
    static bool endpoint_error_handling = true;
    _endpoint = _listener->createEndpointFromConnRequest(conn_request, endpoint_error_handling);
  }

  void releaseEndpoint() { _endpoint.reset(); }

};

static void listener_cb(ucp_conn_request_h conn_request, void* arg)
{
  ucp_conn_request_attr_t attr{};
  ListenerCTX* listener_ctx = reinterpret_cast<ListenerCTX*>(arg);

  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
  ucp_conn_request_query(conn_request, &attr);

  if (listener_ctx->isAvailable()) {
    listener_ctx->createEndpointFromConnRequest(conn_request);
  } else {
    // The server is already handling a connection request from a client,
    // reject this new one
    ucp_listener_reject(listener_ctx->getListener()->getHandle(), conn_request);
  }
}

// 每个node起一个root
// 每个root有一个bootstrapper
// node之内通过cudaGetDeviceCount和cudaGetDevice获取rank
// node之间通过rdma交换address和key

// root作为bootstrapper
// 其他rank作为client
class Bootstrapper {
 public:
 Bootstrapper() {
    auto context = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
    auto worker = context->createWorker();
    auto listener_ctx = std::make_unique<ListenerCTX>(worker);
    auto listener = worker->createListener(listener_port, listener_cb, listener_ctx.get());
    listener_ctx->setListener(listener);
    auto endpoint = worker->createEndpointFromHostname("127.0.0.1", listener_port, true);
    
    while (listener_ctx->isAvailable()) {
      worker->progress();
    }


    

  }

  void initialize() {
    localRank_ = rank_;
    remoteRanks_.resize(totalRanks_);

    for (int i = 0; i < totalRanks_; ++i) {
      if (i == rank_) continue;

      uint64_t remoteAddr = getRemoteAddress(i);
      auto remoteKey = getRemoteKey(i);
      worker_->memPut(&localRank_, sizeof(localRank_), remoteAddr, remoteKey);
      worker_->memGet(&remoteRanks_[i], sizeof(remoteRanks_[i]), remoteAddr, remoteKey);
    }

    worker_->progress();
  }

  void printRanks() {
    std::cout << "Node " << rank_ << " collected ranks: ";
    for (int i = 0; i < totalRanks_; ++i) {
      std::cout << remoteRanks_[i] << " ";
    }
    std::cout << std::endl;
  }

 private:
  int rank_;
  int totalRanks_;
  int localRank_;
  std::vector<int> remoteRanks_;

  std::shared_ptr<ucxx::Context> context_;
  std::shared_ptr<ucxx::Worker> worker_;
  std::shared_ptr<ucxx::RemoteKey> localKey_;

  uint64_t getRemoteAddress(int remoteRank) {
    return reinterpret_cast<uint64_t>(&remoteRanks_[remoteRank]);
  }
  std::shared_ptr<ucxx::RemoteKey> getRemoteKey(int remoteRank) {
    return localKey_;
  }
};

int main() {
  int rank = 0;
  int totalRanks = 4;
  SimpleBootstrap bootstrap(rank, totalRanks);
  bootstrap.initialize();
  bootstrap.printRanks();

  return 0;
}