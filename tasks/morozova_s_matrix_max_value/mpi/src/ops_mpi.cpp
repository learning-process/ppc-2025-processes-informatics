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
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto &matrix = GetInput();
  int rows = 0;
  int cols = 0;
  if (rank == 0) {
    rows = static_cast<int>(matrix.size());
    if (rows > 0) {
      cols = static_cast<int>(matrix[0].size());
      for (int i = 0; i < rows; i++) {
        if (static_cast<int>(matrix[i].size()) != cols) {
          rows = -1;
          cols = -1;
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
  int chunk_size = (rows + size - 1) / size;
  int start_row = rank * chunk_size;
  int end_row = std::min(start_row + chunk_size, rows);
  int local_rows = end_row - start_row;
  int local_elems = local_rows * cols;
  std::vector<int> flat;
  if (rank == 0) {
    flat.resize(rows * cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        flat[i * cols + j] = matrix[i][j];
      }
    }
  }
  std::vector<int> send_counts(size);
  std::vector<int> displacements(size);
  for (int i = 0; i < size; i++) {
    int s_row = i * chunk_size;
    int e_row = std::min(s_row + chunk_size, rows);
    send_counts[i] = (e_row - s_row) * cols;
    displacements[i] = s_row * cols;
  }
  std::vector<int> local_data(local_elems);
  MPI_Scatterv(rank == 0 ? flat.data() : nullptr, send_counts.data(), displacements.data(), MPI_INT, local_data.data(),
               local_elems, MPI_INT, 0, MPI_COMM_WORLD);
  int local_max = std::numeric_limits<int>::min();
  for (int i = 0; i < local_elems; i++) {
    local_max = std::max(local_max, local_data[i]);
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
