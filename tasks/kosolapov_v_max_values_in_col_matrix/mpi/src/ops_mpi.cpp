#include "kosolapov_v_max_values_in_col_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "kosolapov_v_max_values_in_col_matrix/common/include/common.hpp"

namespace kosolapov_v_max_values_in_col_matrix {

KosolapovVMaxValuesInColMatrixMPI::KosolapovVMaxValuesInColMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = {};
}

bool KosolapovVMaxValuesInColMatrixMPI::ValidationImpl() {
  const auto &matrix = GetInput();
  for (size_t i = 0; i < matrix.size() - 1; i++) {
    if (matrix[i].size() != matrix[i + 1].size()) {
      return false;
    }
  }
  return (GetOutput().empty());
}

bool KosolapovVMaxValuesInColMatrixMPI::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().resize(GetInput()[0].size());
  return true;
}

bool KosolapovVMaxValuesInColMatrixMPI::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }
  int processes_count = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &processes_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const int columns = static_cast<int>(matrix[0].size());

  const int columns_per_proc = columns / processes_count;
  const int remainder = columns % processes_count;

  const int start = (rank * columns_per_proc) + std::min(rank, remainder);

  auto local_maxs = CalculateLocalMax(matrix, rank, processes_count, columns);
  std::vector<int> global_maxs;
  if (rank == 0) {
    global_maxs.resize(columns);
    for (size_t i = 0; i < local_maxs.size(); i++) {
      global_maxs[start + i] = local_maxs[i];
    }

    for (int proc = 1; proc < processes_count; proc++) {
      const int proc_start = (proc * columns_per_proc) + std::min(proc, remainder);
      const int proc_end = proc_start + columns_per_proc + (proc < remainder ? 1 : 0);
      const int proc_columns_count = proc_end - proc_start;
      std::vector<int> proc_maxs(proc_columns_count);
      MPI_Recv(proc_maxs.data(), proc_columns_count, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (int i = 0; i < proc_columns_count; i++) {
        global_maxs[proc_start + i] = proc_maxs[i];
      }
    }
    for (int proc = 1; proc < processes_count; proc++) {
      MPI_Send(global_maxs.data(), columns, MPI_INT, proc, 1, MPI_COMM_WORLD);
    }
  } else {
    MPI_Send(local_maxs.data(), static_cast<int>(local_maxs.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);
    global_maxs.resize(columns);
    MPI_Recv(global_maxs.data(), columns, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  GetOutput() = global_maxs;
  return true;
}

bool KosolapovVMaxValuesInColMatrixMPI::PostProcessingImpl() {
  return true;
}

std::vector<int> KosolapovVMaxValuesInColMatrixMPI::CalculateLocalMax(const std::vector<std::vector<int>> &matrix,
                                    int rank, int processes_count, int columns){
  const int columns_per_proc = columns / processes_count;
  const int remainder = columns % processes_count;

  const int start = (rank * columns_per_proc) + std::min(rank, remainder);
  const int end = start + columns_per_proc + (rank < remainder ? 1 : 0);
  std::vector<int> local_maxs(end - start);
  for (int i = start; i < end; i++) {
    int temp_max = matrix[0][i];
    for (const auto& row: matrix) {
      temp_max = std::max(row[i], temp_max);
    }
    local_maxs[i - start] = temp_max;
  }
  return local_maxs;
}
}  // namespace kosolapov_v_max_values_in_col_matrix
