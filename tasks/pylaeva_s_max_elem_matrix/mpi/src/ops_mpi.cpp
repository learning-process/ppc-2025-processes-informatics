#include "pylaeva_s_max_elem_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "pylaeva_s_max_elem_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace pylaeva_s_max_elem_matrix {

PylaevaSMaxElemMatrixMPI::PylaevaSMaxElemMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PylaevaSMaxElemMatrixMPI::ValidationImpl() {
  return (static_cast<size_t>(std::get<0>(GetInput()))==std::get<1>(GetInput()).size()) && (static_cast<size_t>(std::get<0>(GetInput()))>0) && (GetOutput() == 0);
}

bool PylaevaSMaxElemMatrixMPI::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

bool PylaevaSMaxElemMatrixMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int matrix_size = 0;
  std::vector<int> matrix_data;
  
  if (rank == 0) {
    matrix_size = static_cast<int>(std::get<0>(GetInput()));
    matrix_data = std::get<1>(GetInput());
    
    if (matrix_size <= 0 || static_cast<size_t>(matrix_size) != matrix_data.size()) {
      return false;
    }
  }

  MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (matrix_size == 0) {
    GetOutput() = std::numeric_limits<int>::min();
    return true;
  }

  std::vector<int> local_matrix_data(matrix_size);
  if (rank == 0) {
    local_matrix_data = matrix_data;
  }
  MPI_Bcast(local_matrix_data.data(), matrix_size, MPI_INT, 0, MPI_COMM_WORLD);

  int local_size = matrix_size / size;
  int remainder = matrix_size % size;

  int start_idx = 0;
  int end_idx = 0;

  if (rank < remainder) {
    start_idx = rank * (local_size + 1);
    end_idx = start_idx + local_size + 1;
  } else {
    start_idx = remainder * (local_size + 1) + (rank - remainder) * local_size;
    end_idx = start_idx + local_size;
  }

  end_idx = std::min(end_idx, matrix_size);

  int local_max = std::numeric_limits<int>::min();
  
  if (start_idx < end_idx) {
    for (int i = start_idx; i < end_idx; i++) {
      if (local_matrix_data[i] > local_max) {
        local_max = local_matrix_data[i];
      }
    }
  }

  if (rank >= matrix_size) {
    local_max = std::numeric_limits<int>::min();
  }

  int global_max = 0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool PylaevaSMaxElemMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace pylaeva_s_max_elem_matrix
