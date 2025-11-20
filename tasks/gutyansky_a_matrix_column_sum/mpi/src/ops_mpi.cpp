#include "gutyansky_a_matrix_column_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "gutyansky_a_matrix_column_sum/common/include/common.hpp"

namespace gutyansky_a_matrix_column_sum {

GutyanskyAMatrixColumnSumMPI::GutyanskyAMatrixColumnSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  GetInput() = in;
  GetOutput() = {};
}

bool GutyanskyAMatrixColumnSumMPI::ValidationImpl() {
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return GetInput().rows > 0 && GetInput().cols > 0 && GetInput().data.size() == GetInput().rows * GetInput().cols;
  }

  return true;
}

bool GutyanskyAMatrixColumnSumMPI::PreProcessingImpl() {
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput().resize(GetInput().cols);
    return GetOutput().size() == GetInput().cols;
  }

  return true;
}

bool GutyanskyAMatrixColumnSumMPI::RunImpl() {
  int rank = -1;
  int p_count = -1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p_count);

  size_t row_count = GetInput().rows;
  size_t col_count = GetInput().cols;

  MPI_Bcast(&row_count, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&col_count, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  if (row_count == 0 || col_count == 0) {
    return false;
  }

  size_t chunk_size = row_count / static_cast<size_t>(p_count);
  size_t remainder_size = row_count % static_cast<size_t>(p_count);
  size_t elements_count = col_count * chunk_size;

  std::vector<int32_t> input_data_chunk(elements_count);
  std::vector<int32_t> partial_res(col_count, static_cast<int32_t>(0));

  MPI_Scatter(GetInput().data.data(), static_cast<int>(elements_count), MPI_INTEGER4, input_data_chunk.data(),
              static_cast<int>(elements_count), MPI_INTEGER4, 0, MPI_COMM_WORLD);

  for (size_t i = 0; i < chunk_size; i++) {
    for (size_t j = 0; j < col_count; j++) {
      partial_res[j] += input_data_chunk[(i * col_count) + j];
    }
  }

  if (rank == 0 && remainder_size > 0) {
    size_t remainder_offset = chunk_size * static_cast<size_t>(p_count);
    for (size_t i = remainder_offset; i < row_count; i++) {
      for (size_t j = 0; j < col_count; j++) {
        partial_res[j] += GetInput().data[(i * col_count) + j];
      }
    }
  }

  MPI_Reduce(partial_res.data(), GetOutput().data(), static_cast<int>(col_count), MPI_INTEGER4, MPI_SUM, 0,
             MPI_COMM_WORLD);

  return true;
}

bool GutyanskyAMatrixColumnSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gutyansky_a_matrix_column_sum
