#include "zenin_a_sum_values_by_columns_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "zenin_a_sum_values_by_columns_matrix/common/include/common.hpp"

namespace zenin_a_sum_values_by_columns_matrix {

ZeninASumValuesByColumnsMatrixMPI::ZeninASumValuesByColumnsMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ZeninASumValuesByColumnsMatrixMPI::ValidationImpl() {
  auto &input = GetInput();
  return ((std::get<0>(input)) * std::get<1>(input) == std::get<2>(input).size() && (GetOutput().empty()));
}

bool ZeninASumValuesByColumnsMatrixMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }
  GetOutput().clear();
  return true;
}

void ZeninASumValuesByColumnsMatrixMPI::CalculateLocalSums(const std::vector<double> &matrix_data, size_t columns,
                                                           size_t total_rows, size_t start_column,
                                                           size_t cols_this_process, std::vector<double> &local_sums) {
  for (size_t local_column = 0; local_column < cols_this_process; ++local_column) {
    size_t global_col = start_column + local_column;
    for (size_t row = 0; row < total_rows; ++row) {
      local_sums[local_column] += matrix_data[(row * columns) + global_col];
    }
  }
}

void ZeninASumValuesByColumnsMatrixMPI::PrepareGathervParameters(int world_size, size_t base_cols_per_process,
                                                                 size_t remain, std::vector<int> &recv_counts,
                                                                 std::vector<int> &displacements) {
  for (int i = 0; i < world_size; ++i) {
    recv_counts[i] = static_cast<int>(base_cols_per_process);
    if (i == world_size - 1) {
      recv_counts[i] += static_cast<int>(remain);
    }
    if (i > 0) {
      displacements[i] = displacements[i - 1] + recv_counts[i - 1];
    }
  }
}

bool ZeninASumValuesByColumnsMatrixMPI::RunImpl() {
  auto &input = GetInput();
  int world_size = 0;
  int rank = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  size_t columns = 0;
  std::vector<double> matrix_data;
  size_t total_rows = 0;

  if (rank == 0) {
    columns = std::get<1>(input);
    matrix_data = std::get<2>(input);
    total_rows = std::get<0>(input);
    if (matrix_data.size() % columns != 0) {
      return false;
    }
  }

  MPI_Bcast(&columns, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&total_rows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (columns == 0) {
    return false;
  }

  size_t base_cols_per_process = columns / world_size;
  size_t remain = columns % world_size;

  size_t start_column = 0;
  size_t cols_this_process = 0;

  if (rank == world_size - 1) {
    start_column = rank * base_cols_per_process;
    cols_this_process = base_cols_per_process + remain;
  } else {
    start_column = rank * base_cols_per_process;
    cols_this_process = base_cols_per_process;
  }

  if (rank != 0) {
    matrix_data.resize(total_rows * columns);
  }

  MPI_Bcast(matrix_data.data(), static_cast<int>(matrix_data.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_sums(cols_this_process, 0.0);
  CalculateLocalSums(matrix_data, columns, total_rows, start_column, cols_this_process, local_sums);

  std::vector<double> global_sums;
  if (rank == 0) {
    global_sums.resize(columns, 0.0);
  }

  std::vector<int> recv_counts(world_size, 0);
  std::vector<int> displacements(world_size, 0);

  if (rank == 0) {
    PrepareGathervParameters(world_size, base_cols_per_process, remain, recv_counts, displacements);
  }

  MPI_Gatherv(local_sums.data(), static_cast<int>(local_sums.size()), MPI_DOUBLE, global_sums.data(),
              recv_counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    global_sums.resize(columns);
  }

  MPI_Bcast(global_sums.data(), static_cast<int>(columns), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = global_sums;

  return true;
}

bool ZeninASumValuesByColumnsMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zenin_a_sum_values_by_columns_matrix
