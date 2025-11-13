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
  const auto& matrix_data = std::get<1>(GetInput());
  const auto& matrix_size = static_cast<int>(std::get<0>(GetInput()));

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int local_size = matrix_size / size;
  int remainder = matrix_size % size;

  int start_idx = 0;
  int end_idx = 0;

  if (rank < remainder) {
    start_idx = rank * (local_size + 1);
    end_idx = start_idx + local_size + 1;
  } else {
    start_idx = (remainder * (local_size + 1)) + ((rank - remainder) * local_size);
    end_idx = start_idx + local_size;
  }

  int local_max = INT_MIN;
  
  for (int i = start_idx; i < end_idx && i < matrix_size; i++) {
    if (matrix_data[i] > local_max) {
      local_max = matrix_data[i];
    }
  }

  if (local_size == 0 && rank >= matrix_size) {
    local_max = INT_MIN;
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
