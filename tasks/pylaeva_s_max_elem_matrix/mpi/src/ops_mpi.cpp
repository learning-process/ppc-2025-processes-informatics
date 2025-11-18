#include "pylaeva_s_max_elem_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>  // для std::max
#include <cstddef>    // для size_t
#include <limits>     // для std::numeric_limits

// #include <numeric>
#include <vector>

#include "pylaeva_s_max_elem_matrix/common/include/common.hpp"
// #include "util/include/util.hpp"

namespace pylaeva_s_max_elem_matrix {

PylaevaSMaxElemMatrixMPI::PylaevaSMaxElemMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PylaevaSMaxElemMatrixMPI::ValidationImpl() {
  return (static_cast<size_t>(std::get<0>(GetInput())) == std::get<1>(GetInput()).size()) &&
         (static_cast<size_t>(std::get<0>(GetInput())) > 0) && (GetOutput() == 0);
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

  int matrix_size = static_cast<int>(std::get<0>(GetInput()));
  std::vector<int> matrix_data = std::get<1>(GetInput());

  int local_size = matrix_size / size;
  int remainder = matrix_size % size;

  int start = (rank * local_size) + std::min(rank, remainder);
  int end = start + local_size + (rank < remainder ? 1 : 0);

  int local_max = std::numeric_limits<int>::min();
  for (int i = start; i < end; i++) {
    local_max = std::max(local_max, matrix_data[i]);
  }

  int global_max = std::numeric_limits<int>::min();
  MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_max, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_max;

  return true;
}

bool PylaevaSMaxElemMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace pylaeva_s_max_elem_matrix
