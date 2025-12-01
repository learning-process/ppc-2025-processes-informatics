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
  if (rank == 0) {
    rows = static_cast<int>(matrix.size());
    if (rows > 0) {
      cols = static_cast<int>(matrix[0].size());
      for (int i = 0; i < rows; i++) {
        if (matrix[i].size() != static_cast<size_t>(cols)) {
          rows = cols = -1;
          break;
        }
      }
    }
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rows <= 0 || cols <= 0) {
    GetOutput() = std::numeric_limits<int>::min();
    return true;
  }
  int chunk = (rows + size - 1) / size;
  int start = rank * chunk;
  int end = std::min(start + chunk, rows);
  int local_rows = end - start;
  int local_elems = local_rows * cols;
  std::vector<int> flat;
  if (rank == 0) {
    flat.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        flat[static_cast<size_t>(i) * cols + j] = matrix[i][j];
      }
    }
  }
  std::vector<int> send_counts(size);
  std::vector<int> displs(size);
  for (int i = 0; i < size; i++) {
    const int s = i * chunk;
    const int e = std::min(s + chunk, rows);
    send_counts[i] = (e - s) * cols;
    displs[i] = s * cols;
  }
  std::vector<int> local;
  if (local_elems > 0) {
    local.resize(local_elems);
  }
  MPI_Scatterv(rank == 0 ? flat.data() : nullptr, send_counts.data(), displs.data(), MPI_INT,
               local_elems > 0 ? local.data() : nullptr, local_elems, MPI_INT, 0, MPI_COMM_WORLD);
  int local_max = std::numeric_limits<int>::min();
  for (int v : local) {
    local_max = std::max(local_max, v);
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
