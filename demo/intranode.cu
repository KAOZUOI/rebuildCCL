#include <cstdint>
#include <cstring>
#include <mscclpp/concurrency_device.hpp>

#include "common.hpp"

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__constant__ DeviceHandle<mscclpp::PortChannel> constPortChans[16];
__device__ mscclpp::DeviceSyncer deviceSyncer;
void* localRecvBuff;
void* localSendBuff;

__device__ void localAlltoall(int rank, int nRanksPerNode, size_t nElements) {
  int remoteRank = ((int)blockIdx.x < rank) ? blockIdx.x : blockIdx.x + 1;
  for (int i = 1; i < nRanksPerNode; i++) {
    DeviceHandle<mscclpp::PortChannel> portChan = constPortChans[blockIdx.x];
    if (threadIdx.x == 0 && remoteRank % nRanksPerNode == (rank + i) % nRanksPerNode) {
      portChan.putWithSignalAndFlush(rank * nElements * sizeof(int), remoteRank * nElements * sizeof(int),
                                     nElements * sizeof(int));
    }
    // wait for the data from GPU (rank-i) % nranksPerNode to arrive
    if (threadIdx.x == 0 && remoteRank % nRanksPerNode == (rank - i + nRanksPerNode) % nRanksPerNode) {
      portChan.wait();
    }
    deviceSyncer.sync(nRanksPerNode - 1);
  }
}

__global__ void __launch_bounds__(1024) alltoall0(int rank, size_t nElements) {
  int remoteRank = ((int)blockIdx.x < rank) ? blockIdx.x : blockIdx.x + 1;
  DeviceHandle<mscclpp::PortChannel> portChan = constPortChans[blockIdx.x];
  if (threadIdx.x == 0) {
    portChan.putWithSignal(rank * nElements * sizeof(int), remoteRank * nElements * sizeof(int),
                           nElements * sizeof(int));
  }

  deviceSyncer.sync(gridDim.x);
  if (threadIdx.x == 0) {
    portChan.flush();
    portChan.wait();
  }
}

__global__ void __launch_bounds__(1024) alltoall1(int rank, int nRanksPerNode, size_t nElements) {
  localAlltoall(rank, nRanksPerNode, nElements);
}