#include "eremin_v_hypercube/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <tuple>

#include "eremin_v_hypercube/common/include/common.hpp"

namespace eremin_v_hypercube {

EreminVHypercubeMPI::EreminVHypercubeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool EreminVHypercubeMPI::ValidationImpl() {
  auto &input = GetInput();
  return (std::get<0>(input) < std::get<1>(input)) && (std::get<2>(input) > 0) && (std::get<2>(input) <= 100000000) &&
         (std::get<0>(input) >= -1e9) && (std::get<0>(input) <= 1e9) && (std::get<1>(input) >= -1e9) &&
         (std::get<1>(input) <= 1e9) && (GetOutput() == 0);
}

bool EreminVHypercubeMPI::PreProcessingImpl() {
  return true;
}

bool EreminVHypercubeMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int ndims = static_cast<int>(std::floor(std::log2(world_size)));
  int hypercube_size = (1 << ndims);

  int color = (world_rank < hypercube_size) ? 0 : MPI_UNDEFINED;
  MPI_Comm cube_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &cube_comm);

  double lower_bound = 0.0, upper_bound = 0.0, final_result = 0.0;
  int steps = 0;

  if (world_rank == 0) {
    auto &input = GetInput();
    lower_bound = std::get<0>(input);
    upper_bound = std::get<1>(input);
    steps = std::get<2>(input);
  }

  if (cube_comm != MPI_COMM_NULL) {
    int cube_rank = 0;
    MPI_Comm_rank(cube_comm, &cube_rank);

    for (int d = ndims - 1; d >= 0; --d) {
      int neighbor = cube_rank ^ (1 << d);
      if ((cube_rank & ((1 << d) - 1)) == 0) {
        if ((cube_rank & (1 << d)) == 0) {
          MPI_Send(&lower_bound, 1, MPI_DOUBLE, neighbor, 0, cube_comm);
          MPI_Send(&upper_bound, 1, MPI_DOUBLE, neighbor, 1, cube_comm);
          MPI_Send(&steps, 1, MPI_INT, neighbor, 2, cube_comm);
        } else {
          MPI_Recv(&lower_bound, 1, MPI_DOUBLE, neighbor, 0, cube_comm, MPI_STATUS_IGNORE);
          MPI_Recv(&upper_bound, 1, MPI_DOUBLE, neighbor, 1, cube_comm, MPI_STATUS_IGNORE);
          MPI_Recv(&steps, 1, MPI_INT, neighbor, 2, cube_comm, MPI_STATUS_IGNORE);
        }
      }
    }

    const auto in_function = std::get<3>(GetInput());
    double step_size = (upper_bound - lower_bound) / static_cast<double>(steps);
    double local_result = 0.0;

    for (int i = cube_rank; i < steps; i += hypercube_size) {
      local_result += in_function(lower_bound + ((static_cast<double>(i) + 0.5) * step_size));
    }
    local_result *= step_size;

    double current_sum = local_result;
    for (int d = 0; d < ndims; ++d) {
      int neighbor = cube_rank ^ (1 << d);
      double received_sum = 0.0;
      MPI_Status status;
      MPI_Sendrecv(&current_sum, 1, MPI_DOUBLE, neighbor, 10, &received_sum, 1, MPI_DOUBLE, neighbor, 10, cube_comm,
                   &status);
      current_sum += received_sum;
    }
    final_result = current_sum;

    MPI_Comm_free(&cube_comm);
  }

  MPI_Bcast(&final_result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = final_result;
  return true;
}

bool EreminVHypercubeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace eremin_v_hypercube
