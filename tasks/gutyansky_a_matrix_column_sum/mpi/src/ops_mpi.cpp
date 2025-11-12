#include "gutyansky_a_matrix_column_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gutyansky_a_matrix_column_sum/common/include/common.hpp"

namespace gutyansky_a_matrix_column_sum {

GutyanskyAMatrixColumnSumMPI::GutyanskyAMatrixColumnSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  GetInput() = in;
  GetOutput() = {};
}

bool GutyanskyAMatrixColumnSumMPI::ValidationImpl() {
  return GetInput().rows > 0 && GetInput().cols > 0 && GetInput().data.size() == GetInput().rows * GetInput().cols;
}

bool GutyanskyAMatrixColumnSumMPI::PreProcessingImpl() {
  GetOutput().resize(GetInput().cols);

  return GetOutput().size() == GetInput().cols;
}

bool GutyanskyAMatrixColumnSumMPI::RunImpl() {
  if (GetInput().rows == 0 || GetInput().cols == 0) {
    return false;
  }

  int rank = -1;
  int p_count = -1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p_count);

  size_t row_count = GetInput().rows;
  size_t col_count = GetInput().cols;

  size_t rows_chunk_size = row_count / p_count;
  size_t remainder_size = row_count % p_count;

  size_t start_row_index = (rows_chunk_size * rank) + std::min(static_cast<size_t>(rank), remainder_size);
  size_t end_row_index = start_row_index + rows_chunk_size + (static_cast<size_t>(rank) < remainder_size ? 1 : 0);

  std::vector<int64_t> partial_res(col_count, 0.0);

  for (size_t i = start_row_index; i < end_row_index; i++) {
    for (size_t j = 0; j < col_count; j++) {
      partial_res[j] += GetInput().data[(i * col_count) + j];
    }
  }

  MPI_Allreduce(partial_res.data(), GetOutput().data(), static_cast<int>(col_count), MPI_INTEGER8, MPI_SUM,
                MPI_COMM_WORLD);

  return true;
}

bool GutyanskyAMatrixColumnSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gutyansky_a_matrix_column_sum
