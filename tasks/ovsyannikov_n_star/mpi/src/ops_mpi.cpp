#include "ovsyannikov_n_star/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "ovsyannikov_n_star/common/include/common.hpp"

namespace ovsyannikov_n_star {

OvsyannikovNStarMPI::OvsyannikovNStarMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool OvsyannikovNStarMPI::ValidationImpl() {
  return GetInput().size() == 3;
}

bool OvsyannikovNStarMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool OvsyannikovNStarMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int src = GetInput()[0];
  int dst = GetInput()[1];
  int val = GetInput()[2];

  if (size <= 1 || src >= size || dst >= size) {
    GetOutput() = val;
    return true;
  }

  int res = 0;
  if (src == dst) {
    if (rank == src) {
      res = val;
    }
  } else if (src == 0 || dst == 0) {
    if (rank == src) {
      MPI_Send(&val, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
    }
    if (rank == dst) {
      MPI_Recv(&res, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  } else {
    if (rank == src) {
      MPI_Send(&val, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
      int tmp = 0;
      MPI_Recv(&tmp, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&tmp, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
    }
    if (rank == dst) {
      MPI_Recv(&res, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  GetOutput() = res;
  MPI_Bcast(&GetOutput(), 1, MPI_INT, dst, MPI_COMM_WORLD);
  return true;
}

bool OvsyannikovNStarMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ovsyannikov_n_star
