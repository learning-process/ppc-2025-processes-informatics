#include "morozova_s_matrix_max_value/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "morozova_s_matrix_max_value/common/include/common.hpp"

namespace morozova_s_matrix_max_value {

MorozovaSMatrixMaxValueMPI::MorozovaSMatrixMaxValueMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MorozovaSMatrixMaxValueMPI::ValidationImpl() {
  return !GetInput().empty() && !GetInput()[0].empty() && (GetOutput() == 0);
}

bool MorozovaSMatrixMaxValueMPI::PreProcessingImpl() {
  GetOutput() = GetInput()[0][0];
  return true;
}

bool MorozovaSMatrixMaxValueMPI::RunImpl() {
  const auto &matrix = GetInput();
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int rows = static_cast<int>(matrix.size());
  const int cols = static_cast<int>(matrix[0].size());
  const int rows_per_process = rows / size;
  const int remainder = rows % size;
  const int start_row = (rank * rows_per_process) + std::min(rank, remainder);
  const int end_row = start_row + rows_per_process + ((rank < remainder) ? 1 : 0);
  int local_max = std::numeric_limits<int>::min();
  for (int i = start_row; i < end_row; i++) {
    for (int j = 0; j < cols; j++) {
      local_max = std::max(local_max, matrix[i][j]);
    }
  }
  int global_max = std::numeric_limits<int>::min();
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  GetOutput() = global_max;
  return true;
}

bool MorozovaSMatrixMaxValueMPI::PostProcessingImpl() {
  return GetOutput() != std::numeric_limits<int>::min();
}

}  // namespace morozova_s_matrix_max_value
