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
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto &matrix = GetInput();
  int rows = 0;
  int cols = 0;
  bool valid_matrix = true;
  if (rank == 0) {
    rows = static_cast<int>(matrix.size());
    if (rows > 0) {
      cols = static_cast<int>(matrix[0].size());
      for (int i = 1; i < rows; ++i) {
        if (static_cast<int>(matrix[i].size()) != cols) {
          valid_matrix = false;
          break;
        }
      }
    }
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int valid_int = valid_matrix ? 1 : 0;
  MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  valid_matrix = (valid_int == 1);
  const bool invalid_input = !valid_matrix || rows <= 0 || cols <= 0;
  if (invalid_input) {
    GetOutput() = std::numeric_limits<int>::min();
    return true;
  }
  const int total_elements = rows * cols;
  std::vector<int> all_data;
  if (rank == 0) {
    all_data.resize(total_elements);
    for (int i = 0; i < rows; ++i) {
      std::copy(matrix[i].begin(), matrix[i].end(), all_data.begin() + i * cols);
    }
  }
  const int base_chunk_size = total_elements / size;
  const int remainder = total_elements % size;
  std::vector<int> send_counts(size);
  std::vector<int> displs(size);
  int current_displ = 0;
  for (int i = 0; i < size; ++i) {
    send_counts[i] = base_chunk_size + (i < remainder ? 1 : 0);
    displs[i] = current_displ;
    current_displ += send_counts[i];
  }
  const int local_size = send_counts[rank];
  std::vector<int> local_data(local_size);
  MPI_Scatterv(all_data.data(), send_counts.data(), displs.data(), MPI_INT, local_data.data(), local_size, MPI_INT, 0,
               MPI_COMM_WORLD);
  int local_max = std::numeric_limits<int>::min();
  for (int value : local_data) {
    if (value > local_max) {
      local_max = value;
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
