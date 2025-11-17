#include "morozova_s_matrix_max_value/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "morozova_s_matrix_max_value/common/include/common.hpp"
#include "util/include/util.hpp"

namespace morozova_s_matrix_max_value {

MorozovaSMatrixMaxValueMPI::MorozovaSMatrixMaxValueMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MorozovaSMatrixMaxValueMPI::ValidationImpl() {
  return !GetInput().empty() && !GetInput()[0].empty() && (GetOutput() == 0);
}

bool MorozovaSMatrixMaxValueMPI::PreProcessingImpl() {
  GetOutput() = GetInput()[0][0];
  return true;
}

bool MorozovaSMatrixMaxValueMPI::RunImpl() {
  auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = matrix.size();
  int cols = matrix[0].size();
  int total_elements = rows * cols;

  int elements_per_process = total_elements / size;
  int remainder = total_elements % size;

  int start_element = rank * elements_per_process + std::min(rank, remainder);
  int end_element = start_element + elements_per_process + (rank < remainder ? 1 : 0);

  int local_max = std::numeric_limits<int>::min();

  for (int i = start_element; i < end_element; ++i) {
    int row = i / cols;
    int col = i % cols;
    if (row < rows && col < cols) {
      local_max = std::max(local_max, matrix[row][col]);
    }
  }

  int global_max = 0;

  if (rank == 0) {
    global_max = local_max;
    for (int i = 1; i < size; ++i) {
      int received_max;
      MPI_Recv(&received_max, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      global_max = std::max(global_max, received_max);
    }
  } else {
    MPI_Send(&local_max, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Bcast(&global_max, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool MorozovaSMatrixMaxValueMPI::PostProcessingImpl() {
  return GetOutput() != std::numeric_limits<int>::min();
}

}  // namespace morozova_s_matrix_max_value
