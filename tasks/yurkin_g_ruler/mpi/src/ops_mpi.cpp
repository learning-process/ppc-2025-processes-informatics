#include "yurkin_g_ruler/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>

#include "yurkin_g_ruler/common/include/common.hpp"

namespace yurkin_g_ruler {

YurkinGRulerMPI::YurkinGRulerMPI(const InType &in) {
  SetTypeOfTask(YurkinGRulerMPI::GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinGRulerMPI::ValidationImpl() {
  return (GetInput() >= 0) && (GetOutput() == 0);
}

bool YurkinGRulerMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YurkinGRulerMPI::RunImpl() {
  MPI_Comm topo_comm = MPI_COMM_NULL;
  MPI_Comm_dup(MPI_COMM_WORLD, &topo_comm);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(topo_comm, &rank);
  MPI_Comm_size(topo_comm, &size);

  if (size <= 0) {
    if (topo_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&topo_comm);
    }
    return false;
  }

  const int input_value = GetInput();
  const int src = input_value % size;
  const int dst = (input_value / size) % size;
  const int payload = input_value;

  if (src == dst) {
    if (rank == dst) {
      GetOutput() = payload;
    }
    MPI_Barrier(topo_comm);
    if (topo_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&topo_comm);
    }
    return true;
  }

  const int low = std::min(src, dst);
  const int high = std::max(src, dst);

  if (rank < low || rank > high) {
    MPI_Barrier(topo_comm);
    if (topo_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&topo_comm);
    }
    return true;
  }

  const int direction = (dst > src) ? +1 : -1;

  if (rank == src) {
    const int next = rank + direction;
    MPI_Send(&payload, 1, MPI_INT, next, 0, topo_comm);
    MPI_Barrier(topo_comm);
    if (topo_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&topo_comm);
    }
    return true;
  }

  const int prev = rank - direction;
  int recv_val = 0;
  MPI_Recv(&recv_val, 1, MPI_INT, prev, 0, topo_comm, MPI_STATUS_IGNORE);

  if (rank == dst) {
    GetOutput() = recv_val;
    MPI_Barrier(topo_comm);
    if (topo_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&topo_comm);
    }
    return true;
  }

  const int next = rank + direction;
  MPI_Send(&recv_val, 1, MPI_INT, next, 0, topo_comm);

  MPI_Barrier(topo_comm);
  if (topo_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&topo_comm);
  }
  return true;
}

bool YurkinGRulerMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace yurkin_g_ruler
