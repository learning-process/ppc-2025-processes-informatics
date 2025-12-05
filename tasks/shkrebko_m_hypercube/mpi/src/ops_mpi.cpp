#include "shkrebko_m_hypercube/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "shkrebko_m_hypercube/common/include/common.hpp"

namespace shkrebko_m_hypercube {

namespace {

std::vector<int> IntToBin(int number, int padding = 0) {
  std::vector<int> result;
  while (number > 0) {
    result.push_back(number % 2);
    number = number / 2;
  }
  while (result.size() < static_cast<size_t>(padding)) {
    result.push_back(0);
  }
  std::reverse(result.begin(), result.end());
  return result;
}

int BinToInt(const std::vector<int> &binary) {
  int result = 0;
  std::vector<int> temp = binary;
  std::reverse(temp.begin(), temp.end());
  for (size_t i = 0; i < temp.size(); i++) {
    result += (temp[i] * (1 << i));
  }
  return result;
}

int CalcNextRank(int current_rank, int destination_rank, int world_size) {
  if (current_rank == destination_rank) {
    return current_rank;
  }
  int max_address_length = static_cast<int>(IntToBin(world_size - 1).size());
  std::vector<int> src_bin = IntToBin(current_rank, max_address_length);
  std::vector<int> dest_bin = IntToBin(destination_rank, max_address_length);
  for (size_t i = 0; i < dest_bin.size(); i++) {
    if (src_bin[i] != dest_bin[i]) {
      src_bin[i] = dest_bin[i];
      break;
    }
  }
  int next_rank = BinToInt(src_bin);
  return (next_rank < world_size) ? next_rank : destination_rank;
}

void SendHypercubeData(const HypercubeData &data, int dest_rank, int base_tag) {
  MPI_Send(&data.value, 1, MPI_INT, dest_rank, base_tag, MPI_COMM_WORLD);
  MPI_Send(&data.destination, 1, MPI_INT, dest_rank, base_tag + 1, MPI_COMM_WORLD);
  MPI_Send(&data.finish, 1, MPI_C_BOOL, dest_rank, base_tag + 2, MPI_COMM_WORLD);
  int path_size = static_cast<int>(data.path.size());
  MPI_Send(&path_size, 1, MPI_INT, dest_rank, base_tag + 3, MPI_COMM_WORLD);
  if (path_size > 0) {
    MPI_Send(data.path.data(), path_size, MPI_INT, dest_rank, base_tag + 4, MPI_COMM_WORLD);
  }
}

void RecvHypercubeData(HypercubeData &data, int src_rank, int base_tag, MPI_Status *status = MPI_STATUS_IGNORE) {
  MPI_Recv(&data.value, 1, MPI_INT, src_rank, base_tag, MPI_COMM_WORLD, status);
  MPI_Recv(&data.destination, 1, MPI_INT, src_rank, base_tag + 1, MPI_COMM_WORLD, status);
  MPI_Recv(&data.finish, 1, MPI_C_BOOL, src_rank, base_tag + 2, MPI_COMM_WORLD, status);
  int path_size;
  MPI_Recv(&path_size, 1, MPI_INT, src_rank, base_tag + 3, MPI_COMM_WORLD, status);
  data.path.resize(path_size);
  if (path_size > 0) {
    MPI_Recv(data.path.data(), path_size, MPI_INT, src_rank, base_tag + 4, MPI_COMM_WORLD, status);
  }
}

void BroadcastResult(HypercubeData &result, int world_rank, int world_size) {
  if (world_rank == 0) {
    for (int i = 1; i < world_size; i++) {
      SendHypercubeData(result, i, 100);
    }
  } else {
    RecvHypercubeData(result, 0, 100);
  }
}

void ProcessRoot(HypercubeData &local_data, int world_size) {
  local_data.path.push_back(0);
  local_data.finish = false;

  if (local_data.destination != 0) {
    int next_rank = CalcNextRank(0, local_data.destination, world_size);
    SendHypercubeData(local_data, next_rank, 0);
    RecvHypercubeData(local_data, MPI_ANY_SOURCE, 50);
  } else {
    local_data.finish = true;
  }
  HypercubeData finish_data;
  finish_data.finish = true;
  for (int i = 0; i < world_size; i++) {
    if (std::find(local_data.path.begin(), local_data.path.end(), i) == local_data.path.end()) {
      SendHypercubeData(finish_data, i, 0);
    }
  }
}

void ProcessIntermediate(HypercubeData &local_data, int world_rank, int world_size) {
  if (!local_data.finish) {
    local_data.path.push_back(world_rank);

    if (world_rank == local_data.destination) {
      local_data.finish = true;
      SendHypercubeData(local_data, 0, 50);
    } else {
      int next_rank = CalcNextRank(world_rank, local_data.destination, world_size);
      SendHypercubeData(local_data, next_rank, 0);
    }
  }
}

}  // namespace

ShkrebkoMHypercubeMPI::ShkrebkoMHypercubeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = HypercubeData();
}

bool ShkrebkoMHypercubeMPI::ValidationImpl() {
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if ((world_size & (world_size - 1)) != 0) {
    return false;
  }
  if (GetInput().size() < 2) {
    return false;
  }
  int destination = GetInput()[1];
  if (destination < 0 || destination >= world_size) {
    return false;
  }
  if (GetInput()[0] <= 0) {
    return false;
  }
  return true;
}

bool ShkrebkoMHypercubeMPI::PreProcessingImpl() {
  if (GetInput().size() >= 2) {
    GetOutput().value = GetInput()[0];
    GetOutput().destination = GetInput()[1];
  } else {
    GetOutput().value = 1;
    GetOutput().destination = 0;
  }
  return true;
}

bool ShkrebkoMHypercubeMPI::RunImpl() {
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  HypercubeData local_data;
  if (world_rank == 0) {
    local_data.value = GetOutput().value;
    local_data.destination = GetOutput().destination;
    ProcessRoot(local_data, world_size);
  } else {
    MPI_Status status;
    RecvHypercubeData(local_data, MPI_ANY_SOURCE, 0, &status);
    ProcessIntermediate(local_data, world_rank, world_size);
  }
  BroadcastResult(local_data, world_rank, world_size);
  GetOutput() = local_data;
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool ShkrebkoMHypercubeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkrebko_m_hypercube
