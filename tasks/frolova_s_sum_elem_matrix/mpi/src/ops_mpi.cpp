#include "frolova_s_sum_elem_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <numeric>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

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

  for (const auto &row : matrix) {
    if (row.size() != cols) {
      return false;
    }
  }

  return true;
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

  const int base_rows = rows / size;
  const int remainder = rows % size;

  const int my_rows = base_rows + (rank < remainder ? 1 : 0);
  const int start_row = rank * base_rows + std::min(rank, remainder);
  const int end_row = start_row + my_rows;

  long long local_sum = 0;

  for (int i = start_row; i < end_row; ++i) {
    local_sum += std::accumulate(matrix[i].begin(), matrix[i].end(), 0LL);
  }

  long long global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_sum;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool FrolovaSSumElemMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace frolova_s_sum_elem_matrix


