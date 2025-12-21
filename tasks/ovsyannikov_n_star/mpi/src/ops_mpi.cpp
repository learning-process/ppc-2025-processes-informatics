#include "ovsyannikov_n_star/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

namespace ovsyannikov_n_star {

OvsyannikovNStarMPI::OvsyannikovNStarMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool OvsyannikovNStarMPI::ValidationImpl() {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto &in = GetInput();
  return in.size() == 3 && in[0] >= 0 && in[0] < size && in[1] >= 0 && in[1] < size;
}

bool OvsyannikovNStarMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool OvsyannikovNStarMPI::RunImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int src = GetInput()[0];
  const int dst = GetInput()[1];
  const int val = GetInput()[2];
  const int center = 0;
  int res = 0;

  if (src == dst) {
    if (rank == src) {
      res = val;
    }
  } else if (src == center || dst == center) {
    if (rank == src) {
      MPI_Send(&val, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
    } else if (rank == dst) {
      MPI_Recv(&res, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  } else {
    // Маршрут через центр: src -> 0 -> dst
    if (rank == src) {
      MPI_Send(&val, 1, MPI_INT, center, 0, MPI_COMM_WORLD);
    } else if (rank == center) {
      int buf;
      MPI_Recv(&buf, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&buf, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
    } else if (rank == dst) {
      MPI_Recv(&res, 1, MPI_INT, center, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
