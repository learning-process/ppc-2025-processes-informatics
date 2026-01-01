#include "agafonov_i_torus_grid/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>

#include "agafonov_i_torus_grid/common/include/common.hpp"

namespace agafonov_i_torus_grid {

TorusGridTaskMPI::TorusGridTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TorusGridTaskMPI::ValidationImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  auto data = GetInput();
  if (data.source_rank < 0 || data.source_rank >= world_size || data.dest_rank < 0 || data.dest_rank >= world_size) {
    return false;
  }

  int dims = static_cast<int>(std::sqrt(world_size));
  return dims * dims == world_size;
}

bool TorusGridTaskMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool TorusGridTaskMPI::RunImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  auto data = GetInput();
  int res = 0;

  if (data.source_rank == data.dest_rank) {
    if (world_rank == data.source_rank) {
      res = data.value;
    }
  } else {
    if (world_rank == data.source_rank) {
      MPI_Send(&data.value, 1, MPI_INT, data.dest_rank, 0, MPI_COMM_WORLD);
    } else if (world_rank == data.dest_rank) {
      MPI_Recv(&res, 1, MPI_INT, data.source_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  MPI_Bcast(&res, 1, MPI_INT, data.dest_rank, MPI_COMM_WORLD);

  GetOutput() = res;

  return true;
}

bool TorusGridTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace agafonov_i_torus_grid
