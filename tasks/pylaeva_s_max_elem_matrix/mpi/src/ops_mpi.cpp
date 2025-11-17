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

  int local_size = matrix_size / size;
  int remainder = matrix_size % size;

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  for (int i = 0; i < size; i++) {
    sendcounts[i] = (i < remainder) ? local_size + 1 : local_size;
    displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
  }

  std::vector<int> local_data(sendcounts[rank]);

  if (rank == 0) {
    MPI_Scatterv(matrix_data.data(), sendcounts.data(), displs.data(), MPI_INT, local_data.data(), sendcounts[rank],
                 MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, sendcounts.data(), displs.data(), MPI_INT, local_data.data(), sendcounts[rank], MPI_INT, 0,
                 MPI_COMM_WORLD);
  }

  int local_max = std::numeric_limits<int>::min();
  for (int val : local_data) {
    local_max = std::max(val, local_max);
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
