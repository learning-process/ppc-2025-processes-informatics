#include "morozova_s_broadcast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>

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
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size <= 0) {
    return false;
  }

  if (root_ < 0 || root_ >= size) {
    return false;
  }

  if (!GetOutput().empty()) {
    return false;
  }

  return true;
}

bool MorozovaSBroadcastMPI::PreProcessingImpl() {
  int dummy = 0;
  CustomBroadcast(&dummy, 1, MPI_INT, root_, MPI_COMM_WORLD);
  return true;
}

void MorozovaSBroadcastMPI::CustomBroadcast(void *data, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  for (int src = 0; src < size; ++src) {
    if (rank == src && src != root) {
      MPI_Send(data, count, datatype, root, 0, comm);
    }
    if (rank == root && src != root) {
      MPI_Recv(data, count, datatype, src, 0, comm, MPI_STATUS_IGNORE);
    }
  }

  for (int dst = 0; dst < size; ++dst) {
    if (rank == root && dst != root) {
      MPI_Send(data, count, datatype, dst, 1, comm);
    }
    if (rank == dst && dst != root) {
      MPI_Recv(data, count, datatype, root, 1, comm, MPI_STATUS_IGNORE);
    }
  }
}

bool MorozovaSBroadcastMPI::RunImpl() {
  int data_size = static_cast<int>(GetInput().size());

  CustomBroadcast(&data_size, 1, MPI_INT, root_, MPI_COMM_WORLD);

  GetOutput().resize(data_size);

  if (data_size > 0) {
    std::copy(GetInput().begin(), GetInput().end(), GetOutput().begin());
    CustomBroadcast(GetOutput().data(), data_size, MPI_INT, root_, MPI_COMM_WORLD);
  }

  return true;
}

bool MorozovaSBroadcastMPI::PostProcessingImpl() {
  return true;
}

}  // namespace morozova_s_broadcast
