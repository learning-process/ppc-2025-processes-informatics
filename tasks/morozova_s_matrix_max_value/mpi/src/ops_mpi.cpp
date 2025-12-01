#include "morozova_s_matrix_max_value/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
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
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto &matrix = GetInput();
  int rows = 0, cols = 0;
  bool valid_matrix = true;
  if (rank == 0) {
    rows = static_cast<int>(matrix.size());
    cols = rows > 0 ? static_cast<int>(matrix[0].size()) : 0;

    for (int i = 1; i < rows && valid_matrix; ++i) {
      if (matrix[i].size() != static_cast<size_t>(cols)) {
        valid_matrix = false;
      }
    }
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int valid_int = valid_matrix ? 1 : 0;
  MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  valid_matrix = (valid_int == 1);
  if (!valid_matrix || rows <= 0 || cols <= 0) {
    GetOutput() = std::numeric_limits<int>::min();
    return true;
  }
  const int total_elements = rows * cols;
  std::vector<int> all_data;
  if (rank == 0) {
    all_data.reserve(total_elements);
    for (int i = 0; i < rows; ++i) {
      all_data.insert(all_data.end(), matrix[i].begin(), matrix[i].end());
    }
  }
  const int base_chunk = total_elements / size;
  const int remainder = total_elements % size;
  std::vector<int> send_counts(size), displs(size);
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    send_counts[i] = base_chunk + (i < remainder ? 1 : 0);
    displs[i] = offset;
    offset += send_counts[i];
  }
  const int local_size = send_counts[rank];
  std::vector<int> local_data(local_size);
  MPI_Scatterv(rank == 0 ? all_data.data() : nullptr, send_counts.data(), displs.data(), MPI_INT, local_data.data(),
               local_size, MPI_INT, 0, MPI_COMM_WORLD);
  int local_max = std::numeric_limits<int>::min();
  for (int val : local_data) {
    local_max = std::max(local_max, val);
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
