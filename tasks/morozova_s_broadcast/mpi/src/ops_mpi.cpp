#include "morozova_s_broadcast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "morozova_s_broadcast/common/include/common.hpp"

namespace morozova_s_broadcast {

MorozovaSBroadcastMPI::MorozovaSBroadcastMPI(const InType &in) : BaseTask(), root_(0) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

MorozovaSBroadcastMPI::MorozovaSBroadcastMPI(const InType &in, int root) : BaseTask(), root_(root) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool MorozovaSBroadcastMPI::ValidationImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size <= 0) {
    return false;
  }
  if (root_ < 0 || root_ >= size) {
    return false;
  }
  if (rank == root_ && GetInput().empty()) {
    return false;
  }
  if (!GetOutput().empty()) {
    return false;
  }
  return true;
}

bool MorozovaSBroadcastMPI::PreProcessingImpl() {
  return true;
}

void MorozovaSBroadcastMPI::CustomBroadcast(void *data, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  int virtual_rank = (rank - root + size) % size;
  for (int step = 1; step < size; step <<= 1) {
    if (virtual_rank < step) {
      int dest = virtual_rank + step;
      if (dest < size) {
        int real_dest = (dest + root) % size;
        MPI_Send(data, count, datatype, real_dest, 0, comm);
      }
    } else if (virtual_rank < 2 * step) {
      int src = virtual_rank - step;
      int real_src = (src + root) % size;
      MPI_Recv(data, count, datatype, real_src, 0, comm, MPI_STATUS_IGNORE);
    }
  }
}

bool MorozovaSBroadcastMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int data_size = 0;
  if (rank == root_) {
    data_size = static_cast<int>(GetInput().size());
  }
  CustomBroadcast(&data_size, 1, MPI_INT, root_, MPI_COMM_WORLD);
  if (data_size <= 0) {
    return false;
  }
  GetOutput().resize(data_size);
  if (rank == root_) {
    std::copy(GetInput().begin(), GetInput().end(), GetOutput().begin());
  }
  CustomBroadcast(GetOutput().data(), data_size, MPI_INT, root_, MPI_COMM_WORLD);
  return true;
}

bool MorozovaSBroadcastMPI::PostProcessingImpl() {
  return true;
}

}  // namespace morozova_s_broadcast
