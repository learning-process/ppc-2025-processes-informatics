#include "shkrebko_m_hypercube/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "shkrebko_m_hypercube/common/include/common.hpp"

namespace shkrebko_m_hypercube {

namespace {

int GetNextRankCart(int current_rank, int dest_rank, MPI_Comm cart_comm, int ndims) {
  if (current_rank == dest_rank) {
    return current_rank;
  }

  std::vector<int> current_coords(ndims), dest_coords(ndims);
  MPI_Cart_coords(cart_comm, current_rank, ndims, current_coords.data());
  MPI_Cart_coords(cart_comm, dest_rank, ndims, dest_coords.data());

  for (int dim = 0; dim < ndims; ++dim) {
    if (current_coords[dim] != dest_coords[dim]) {
      int direction = dest_coords[dim] - current_coords[dim];
      int source_rank, next_rank;
      MPI_Cart_shift(cart_comm, dim, direction, &source_rank, &next_rank);
      return next_rank;
    }
  }

  return current_rank;
}

void SendHypercubeDataSeq(const HypercubeData &data, int dest_rank, int tag) {
  MPI_Send(&data.value, 1, MPI_INT, dest_rank, tag, MPI_COMM_WORLD);
  MPI_Send(&data.destination, 1, MPI_INT, dest_rank, tag + 1, MPI_COMM_WORLD);
  MPI_Send(&data.finish, 1, MPI_C_BOOL, dest_rank, tag + 2, MPI_COMM_WORLD);
  int path_size = static_cast<int>(data.path.size());
  MPI_Send(&path_size, 1, MPI_INT, dest_rank, tag + 3, MPI_COMM_WORLD);
  if (path_size > 0) {
    MPI_Send(data.path.data(), path_size, MPI_INT, dest_rank, tag + 4, MPI_COMM_WORLD);
  }
}

void RecvHypercubeDataSeq(HypercubeData &data, int src_rank, int tag, MPI_Status *status = MPI_STATUS_IGNORE) {
  MPI_Recv(&data.value, 1, MPI_INT, src_rank, tag, MPI_COMM_WORLD, status);
  MPI_Recv(&data.destination, 1, MPI_INT, src_rank, tag + 1, MPI_COMM_WORLD, status);
  MPI_Recv(&data.finish, 1, MPI_C_BOOL, src_rank, tag + 2, MPI_COMM_WORLD, status);
  int path_size;
  MPI_Recv(&path_size, 1, MPI_INT, src_rank, tag + 3, MPI_COMM_WORLD, status);
  data.path.resize(path_size);
  if (path_size > 0) {
    MPI_Recv(data.path.data(), path_size, MPI_INT, src_rank, tag + 4, MPI_COMM_WORLD, status);
  }
}

}  // namespace

ShkrebkoMHypercubeSEQ::ShkrebkoMHypercubeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = HypercubeData();
}

bool ShkrebkoMHypercubeSEQ::ValidationImpl() {
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (world_size == 1) {
    if (GetInput().size() < 2) {
      return false;
    }
    int destination = GetInput()[1];
    if (destination != 0) {
      return false;
    }
    if (GetInput()[0] <= 0) {
      return false;
    }
    return true;
  }

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

bool ShkrebkoMHypercubeSEQ::PreProcessingImpl() {
  if (GetInput().size() >= 2) {
    GetOutput().value = GetInput()[0];
    GetOutput().destination = GetInput()[1];

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size == 1) {
      GetOutput().destination = 0;
    }
  } else {
    GetOutput().value = 1;
    GetOutput().destination = 0;
  }
  return true;
}

bool ShkrebkoMHypercubeSEQ::RunImpl() {
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (world_size == 1) {
    GetOutput().value = GetOutput().value;
    GetOutput().destination = 0;
    GetOutput().path = {0};
    GetOutput().finish = true;
    return true;
  }

  int ndims = static_cast<int>(log2(world_size));
  std::vector<int> dims(ndims, 2);
  std::vector<int> periods(ndims, 0);
  int reorder = 1;

  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims.data(), periods.data(), reorder, &cart_comm);

  HypercubeData local_data;

  if (world_rank == 0) {
    local_data.value = GetOutput().value;
    local_data.destination = GetOutput().destination;
    local_data.path.push_back(world_rank);
    local_data.finish = false;

    if (local_data.destination != 0) {
      int next_rank = GetNextRankCart(world_rank, local_data.destination, cart_comm, ndims);
      SendHypercubeDataSeq(local_data, next_rank, 0);

      RecvHypercubeDataSeq(local_data, MPI_ANY_SOURCE, 1);
    } else {
      local_data.finish = true;
    }

    // Отправляем завершающие сообщения ТОЛЬКО процессам, не участвовавшим в пути
    HypercubeData finish_msg;
    finish_msg.finish = true;
    finish_msg.value = local_data.value;
    finish_msg.destination = local_data.destination;
    finish_msg.path = local_data.path;

    for (int i = 1; i < world_size; i++) {
      if (std::find(local_data.path.begin(), local_data.path.end(), i) == local_data.path.end()) {
        SendHypercubeDataSeq(finish_msg, i, 0);
      }
    }

    GetOutput() = local_data;
  } else {
    MPI_Status status;
    RecvHypercubeDataSeq(local_data, MPI_ANY_SOURCE, 0, &status);

    if (!local_data.finish) {
      local_data.path.push_back(world_rank);

      if (world_rank == local_data.destination) {
        local_data.finish = true;
        SendHypercubeDataSeq(local_data, 0, 1);
        GetOutput() = local_data;
      } else {
        int next_rank = GetNextRankCart(world_rank, local_data.destination, cart_comm, ndims);
        SendHypercubeDataSeq(local_data, next_rank, 0);
      }
    } else {
      GetOutput() = local_data;
    }
  }

  if (world_size > 1) {
    MPI_Comm_free(&cart_comm);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool ShkrebkoMHypercubeSEQ::PostProcessingImpl() {
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (world_size == 1) {
    return GetOutput().finish && GetOutput().value > 0 && GetOutput().destination == 0 &&
           GetOutput().path == std::vector<int>{0};
  }

  if (GetOutput().path.empty()) {
    return false;
  }
  if (GetOutput().path.front() != 0) {
    return false;
  }
  if (GetOutput().path.back() != GetOutput().destination) {
    return false;
  }
  if (!GetOutput().finish) {
    return false;
  }
  if (GetOutput().value <= 0) {
    return false;
  }

  for (size_t i = 1; i < GetOutput().path.size(); i++) {
    int prev = GetOutput().path[i - 1];
    int curr = GetOutput().path[i];

    int diff = prev ^ curr;
    if ((diff & (diff - 1)) != 0) {
      return false;
    }
  }

  return true;
}

}  // namespace shkrebko_m_hypercube
