#include "morozova_s_matrix_max_value/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "morozova_s_matrix_max_value/common/include/common.hpp"

namespace morozova_s_matrix_max_value {

MorozovaSMatrixMaxValueMPI::MorozovaSMatrixMaxValueMPI(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = 0;
}

bool MorozovaSMatrixMaxValueMPI::ValidationImpl() {
  return true;
}

bool MorozovaSMatrixMaxValueMPI::PreProcessingImpl() {
  return true;
}

bool MorozovaSMatrixMaxValueMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const auto &matrix = GetInput();
  bool is_valid = !matrix.empty() && !matrix[0].empty();
  if (is_valid) {
    const size_t cols = matrix[0].size();
    for (const auto &row : matrix) {
      if (row.size() != cols) {
        is_valid = false;
        break;
      }
    }
  }
  if (!is_valid) {
    GetOutput() = std::numeric_limits<int>::min();
    return true;
  }
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int rows = static_cast<int>(matrix.size());
  int cols_int = static_cast<int>(matrix[0].size());
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<int> flat_matrix(static_cast<size_t>(rows) * cols_int);
  if (rank == 0) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols_int; ++j) {
        flat_matrix[i * cols_int + j] = matrix[i][j];
      }
    }
  }
  MPI_Bcast(flat_matrix.data(), rows * cols_int, MPI_INT, 0, MPI_COMM_WORLD);
  int base_chunk = rows / size;
  int extra = rows % size;
  int start_row = rank * base_chunk + std::min(rank, extra);
  int end_row = start_row + base_chunk + (rank < extra ? 1 : 0);
  end_row = std::min(end_row, rows);
  int local_max = std::numeric_limits<int>::min();
  for (int i = start_row; i < end_row; ++i) {
    for (int j = 0; j < cols_int; ++j) {
      local_max = std::max(local_max, flat_matrix[i * cols_int + j]);
    }
  }
  int global_max = std::numeric_limits<int>::min();
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  GetOutput() = global_max;
  return true;
}

bool MorozovaSMatrixMaxValueMPI::PostProcessingImpl() {
  return true;
}

}  // namespace morozova_s_matrix_max_value
