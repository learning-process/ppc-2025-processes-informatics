#include "egorova_l_find_max_val_col_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "egorova_l_find_max_val_col_matrix/common/include/common.hpp"

namespace egorova_l_find_max_val_col_matrix {

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

EgorovaLFindMaxValColMatrixMPI::EgorovaLFindMaxValColMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>(0);
}

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

bool EgorovaLFindMaxValColMatrixMPI::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return true;
  }

  if (matrix[0].empty()) {
    return true;
  }

  const std::size_t cols = matrix[0].size();
  return std::ranges::all_of(matrix, [cols](const auto &row) { return row.size() == cols; });
}

bool EgorovaLFindMaxValColMatrixMPI::PreProcessingImpl() {
  return true;
}

bool EgorovaLFindMaxValColMatrixMPI::RunImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty() || matrix[0].empty()) {
    GetOutput() = std::vector<int>();
    return true;
  }

  return RunMPIAlgorithm();
}

bool EgorovaLFindMaxValColMatrixMPI::RunMPIAlgorithm() {
  int rank = 0;
  int size = 0;
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

  std::vector<int> flat_matrix = CreateAndBroadcastMatrix(rank, rows, cols);

  const int cols_per_proc = cols / size;
  const int remainder = cols % size;
  int start_col = 0;
  int local_cols_count = 0;

  if (rank < remainder) {
    start_col = rank * (cols_per_proc + 1);
    local_cols_count = cols_per_proc + 1;
  } else {
    start_col = (remainder * (cols_per_proc + 1)) + ((rank - remainder) * cols_per_proc);
    local_cols_count = cols_per_proc;
  }

  std::vector<int> local_max = CalculateLocalMaxima(flat_matrix, rows, cols, start_col, local_cols_count);
  std::vector<int> all_max = GatherResults(local_max, size, cols);

  GetOutput() = all_max;
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

std::vector<int> EgorovaLFindMaxValColMatrixMPI::CreateAndBroadcastMatrix(int rank, int rows, int cols) {
  std::vector<int> flat_matrix(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));

  if (rank == 0) {
    const auto &matrix = GetInput();
    for (int ii = 0; ii < rows; ++ii) {
      for (int jj = 0; jj < cols; ++jj) {
        flat_matrix[(static_cast<std::size_t>(ii) * static_cast<std::size_t>(cols)) + static_cast<std::size_t>(jj)] =
            matrix[ii][jj];
      }
    }
  }

  MPI_Bcast(flat_matrix.data(), rows * cols, MPI_INT, 0, MPI_COMM_WORLD);
  return flat_matrix;
}

std::vector<int> EgorovaLFindMaxValColMatrixMPI::CalculateLocalMaxima(const std::vector<int> &flat_matrix, int rows,
                                                                      int cols, int start_col, int local_cols_count) {
  std::vector<int> local_max(local_cols_count, std::numeric_limits<int>::min());

  for (int local_idx = 0; local_idx < local_cols_count; ++local_idx) {
    const int global_col = start_col + local_idx;
    for (int row = 0; row < rows; ++row) {
      const int value = flat_matrix[(row * cols) + global_col];
      local_max[local_idx] = std::max(value, local_max[local_idx]);
    }
  }

  return local_max;
}

std::vector<int> EgorovaLFindMaxValColMatrixMPI::GatherResults(const std::vector<int> &local_max, int size, int cols) {
  const int cols_per_proc = cols / size;
  const int remainder = cols % size;

  std::vector<int> all_max(cols, std::numeric_limits<int>::min());
  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);

  for (int ii = 0; ii < size; ++ii) {
    recv_counts[ii] = (ii < remainder) ? (cols_per_proc + 1) : cols_per_proc;
    displs[ii] = (ii == 0) ? 0 : displs[ii - 1] + recv_counts[ii - 1];
  }

  MPI_Allgatherv(local_max.data(), static_cast<int>(local_max.size()), MPI_INT, all_max.data(), recv_counts.data(),
                 displs.data(), MPI_INT, MPI_COMM_WORLD);

  return all_max;
}

bool EgorovaLFindMaxValColMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace egorova_l_find_max_val_col_matrix
