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
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int is_valid = 1;
  if (rank == 0) {
    const auto &matrix = GetInput();
    if (matrix.empty() || matrix[0].empty()) {
      is_valid = 0;
    } else {
      const size_t cols = matrix[0].size();
      for (const auto &row : matrix) {
        if (row.size() != cols) {
          is_valid = 0;
          break;
        }
      }
    }
  }
  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return is_valid != 0;
}

bool MorozovaSMatrixMaxValueMPI::PreProcessingImpl() {
  return true;
}

bool MorozovaSMatrixMaxValueMPI::RunImpl() {
  if (!ValidationImpl()) {
    GetOutput() = std::numeric_limits<int>::min();
    return true;
  }
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int rows = 0;
  int cols = 0;
  if (rank == 0) {
    rows = static_cast<int>(GetInput().size());
    cols = static_cast<int>(GetInput()[0].size());
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rows == 0 || cols == 0) {
    GetOutput() = std::numeric_limits<int>::min();
    return true;
  }

  std::vector<int> flat_matrix(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  if (rank == 0) {
    const auto &input = GetInput();
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        flat_matrix[(i * cols) + j] = input[i][j];
      }
    }
  }
  MPI_Bcast(flat_matrix.data(), rows * cols, MPI_INT, 0, MPI_COMM_WORLD);
  int local_max = std::numeric_limits<int>::min();
  int rows_per_proc = rows / size;
  int remainder = rows % size;
  int start_row = 0;
  int end_row = 0;
  if (rank < remainder) {
    start_row = rank * (rows_per_proc + 1);
    end_row = start_row + (rows_per_proc + 1);
  } else {
    start_row = (remainder * (rows_per_proc + 1)) + ((rank - remainder) * rows_per_proc);
    end_row = start_row + rows_per_proc;
  }
  end_row = std::min(end_row, rows);
  for (int i = start_row; i < end_row; ++i) {
    for (int j = 0; j < cols; ++j) {
      const int value = flat_matrix[(i * cols) + j];
      local_max = std::max(local_max, value);
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
