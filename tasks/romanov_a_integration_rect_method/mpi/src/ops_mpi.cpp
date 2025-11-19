#include "romanov_a_integration_rect_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "romanov_a_integration_rect_method/common/include/common.hpp"
#include "util/include/util.hpp"

namespace romanov_a_integration_rect_method {

RomanovAIntegrationRectMethodMPI::RomanovAIntegrationRectMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool RomanovAIntegrationRectMethodMPI::ValidationImpl() {
  if (!IsEqual(GetOutput(), 0.0)) {
    return false;
  } else if (std::get<3>(GetInput()) <= 0) {
    return false;
  } else if (std::get<1>(GetInput()) >= std::get<2>(GetInput())) {
    return false;
  }
  return true;
}

bool RomanovAIntegrationRectMethodMPI::PreProcessingImpl() {
  return true;
}

bool RomanovAIntegrationRectMethodMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_processes = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  double a, b;
  int n;

  double result = 0.0;

  if (rank == 0) {
    a = std::get<1>(GetInput());
    b = std::get<2>(GetInput());
    n = std::get<3>(GetInput());
  }
  MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  auto f = std::get<0>(GetInput());

  int left_border = rank * (n / num_processes);
  int right_border = std::min(n, (rank + 1) * (n / num_processes));

  double delta_x = (b - a) / static_cast<double>(n);
  double mid = a + delta_x * static_cast<double>(left_border) + delta_x / 2.0;

  double current_result = 0.0;

  for (int i = left_border; i < right_border; ++i) {
    current_result += f(mid) * delta_x;
    mid += delta_x;
  }

  MPI_Allreduce(&current_result, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = result;

  return true;
}

bool RomanovAIntegrationRectMethodMPI::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_a_integration_rect_method
