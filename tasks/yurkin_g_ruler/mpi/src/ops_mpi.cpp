#include "yurkin_g_ruler/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "yurkin_g_ruler/common/include/common.hpp"

namespace yurkin_g_ruler {

YurkinGRulerMPI::YurkinGRulerMPI(const InType &in) {
  this->SetTypeOfTask(this->GetStaticTypeOfTask());
  this->GetInput() = in;
  this->GetOutput() = 0;
}

bool YurkinGRulerMPI::ValidationImpl() {
  return (this->GetInput() >= 0) && (this->GetOutput() == 0);
}

bool YurkinGRulerMPI::PreProcessingImpl() {
  this->GetOutput() = 0;
  return true;
}

bool YurkinGRulerMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size <= 0) {
    return false;
  }

  int src = this->GetInput() % size;
  int dst = (this->GetInput() / size) % size;
  src = std::clamp(src, 0, size - 1);
  dst = std::clamp(dst, 0, size - 1);

  int payload = this->GetInput();

  if (src == dst) {
    this->GetOutput() = payload;
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  int next = (rank + 1) % size;
  int prev = (rank - 1 + size) % size;

  if (rank == src) {
    MPI_Send(&payload, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
  } else if (rank == dst) {
    int recv_val = 0;
    MPI_Recv(&recv_val, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    this->GetOutput() = recv_val;
  } else {
    int recv_val = 0;
    MPI_Recv(&recv_val, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(&recv_val, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool YurkinGRulerMPI::PostProcessingImpl() {
  return this->GetOutput() >= 0;
}

}  // namespace yurkin_g_ruler
