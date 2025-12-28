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
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int ndims = static_cast<int>(std::floor(std::log2(size)));
  int hypercube_size = (1 << ndims);

  double lower_bound = 0.0;
  double upper_bound = 0.0;
  int steps = 0;

  if (rank == 0) {
    auto &input = GetInput();
    lower_bound = std::get<0>(input);
    upper_bound = std::get<1>(input);
    steps = std::get<2>(input);
  }
  MPI_Bcast(&lower_bound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&upper_bound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const auto in_function = std::get<3>(GetInput());

  double step_size = (upper_bound - lower_bound) / static_cast<double>(steps);
  double final_result = 0.0;

  if (rank < hypercube_size) {
    double local_result = 0.0;
    for (int i = rank; i < steps; i += hypercube_size) {
      local_result += in_function(lower_bound + ((static_cast<double>(i) + 0.5) * step_size));
    }
    local_result *= step_size;
    double current_sum = local_result;
    for (int d = 0; d < ndims; ++d) {
      int neighbor = rank ^ (1 << d);
      double received_sum = 0.0;
      MPI_Status status;
      MPI_Sendrecv(&current_sum, 1, MPI_DOUBLE, neighbor, 0, &received_sum, 1, MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD,
                   &status);
      current_sum += received_sum;
    }
    final_result = current_sum;
  }

  MPI_Bcast(&final_result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = final_result;
  return true;
}

bool EreminVHypercubeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace eremin_v_hypercube
