#include "frolova_s_sum_elem_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

namespace frolova_s_sum_elem_matrix {

FrolovaSSumElemMatrixMPI::FrolovaSSumElemMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool FrolovaSSumElemMatrixMPI::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  const std::size_t cols = matrix.front().size();
  if (cols == 0) {
    return false;
  }
  return std::ranges::all_of(matrix, [cols](const auto &row) { return row.size() == cols; });
}

bool FrolovaSSumElemMatrixMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool FrolovaSSumElemMatrixMPI::RunImpl() {
  const auto &matrix = GetInput();

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int rows = static_cast<int>(matrix.size());
  const int cols = static_cast<int>(matrix[0].size());

  const int base_rows = rows / size;
  const int remainder = rows % size;

  const int my_rows = base_rows + (rank < remainder ? 1 : 0);
  const int start_row = (rank * base_rows) + std::min(rank, remainder);
  const int end_row = start_row + my_rows;

  int64_t local_sum = 0;

  for (int i = start_row; i < end_row; i++) {
    for (int j = 0; j < cols; j++) {
      local_sum += matrix[i][j];
    }
  }

  int64_t global_sum = 0;

  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_sum;

  return true;
}

bool FrolovaSSumElemMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace frolova_s_sum_elem_matrix
