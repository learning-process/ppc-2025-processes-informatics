#include "romanov_a_integration_rect_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>

#include "romanov_a_integration_rect_method/common/include/common.hpp"

namespace romanov_a_integration_rect_method {

RomanovAIntegrationRectMethodMPI::RomanovAIntegrationRectMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool RomanovAIntegrationRectMethodMPI::ValidationImpl() {
  return (IsEqual(GetOutput(), 0.0) && (std::get<3>(GetInput()) > 0) &&
          (std::get<1>(GetInput()) < std::get<2>(GetInput())));
}

bool RomanovAIntegrationRectMethodMPI::PreProcessingImpl() {
  return true;
}

bool RomanovAIntegrationRectMethodMPI::RunImpl() {
  const auto &[f, a, b, n] = GetInput();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_processes = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  int block_size = (n + num_processes - 1) / num_processes;

  int left_border = rank * block_size;
  int right_border = std::min(n, (rank + 1) * block_size);

  double delta_x = (b - a) / static_cast<double>(n);
  double mid = a + (delta_x * static_cast<double>(left_border)) + (delta_x / 2.0);

  double current_result = 0.0;

  for (int i = left_border; i < right_border; ++i) {
    current_result += f(mid) * delta_x;
    mid += delta_x;
  }

  double result = 0.0;
  MPI_Allreduce(&current_result, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = result;

  return true;
}

bool RomanovAIntegrationRectMethodMPI::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_a_integration_rect_method
