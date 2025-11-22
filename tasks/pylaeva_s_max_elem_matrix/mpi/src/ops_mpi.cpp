#include "pylaeva_s_max_elem_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>  // для std::max
#include <cstddef>    // для size_t
#include <limits>     // для std::numeric_limits
#include <vector>

#include "pylaeva_s_max_elem_matrix/common/include/common.hpp"

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
  }

  MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> sendcounts(size, 0);
  std::vector<int> displs(size, 0);

  int local_size = matrix_size / size;
  int remainder = matrix_size % size;

  if (rank == 0) {
    int offset = 0;
    for (int i = 0; i < size; i++) {
      sendcounts[i] = local_size + (i < remainder ? 1 : 0);
      displs[i] = offset;
      offset += sendcounts[i];
    }
  }

  MPI_Bcast(sendcounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  int local_elements = sendcounts[rank];
  std::vector<int> local_data(local_elements);

  MPI_Scatterv(rank == 0 ? matrix_data.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT, local_data.data(),
               local_elements, MPI_INT, 0, MPI_COMM_WORLD);

  int local_max = std::numeric_limits<int>::min();
  for (int i = 0; i < local_elements; i++) {
    local_max = std::max(local_max, local_data[i]);
  }

  int global_max = std::numeric_limits<int>::min();
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  GetOutput() = global_max;

  return true;
}

bool PylaevaSMaxElemMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace pylaeva_s_max_elem_matrix
