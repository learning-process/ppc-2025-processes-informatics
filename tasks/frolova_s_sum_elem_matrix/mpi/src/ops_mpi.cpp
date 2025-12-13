#include "frolova_s_sum_elem_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdint>
#include <vector>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

namespace frolova_s_sum_elem_matrix {

FrolovaSSumElemMatrixMPI::FrolovaSSumElemMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool FrolovaSSumElemMatrixMPI::ValidationImpl() {
  return true;
}

bool FrolovaSSumElemMatrixMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool FrolovaSSumElemMatrixMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<std::vector<int>> matrix;
  int rows = 0;

  if (rank == 0) {
    matrix = GetInput();
    rows = static_cast<int>(matrix.size());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> row_sizes(rows);
  if (rank == 0) {
    for (int i = 0; i < rows; i++) {
      row_sizes[i] = static_cast<int>(matrix[i].size());
    }
  }
  MPI_Bcast(row_sizes.data(), rows, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> flat_data;
  std::vector<int> displs(rows);

  if (rank == 0) {
    int total_elements = 0;
    for (int i = 0; i < rows; i++) {
      displs[i] = total_elements;
      total_elements += row_sizes[i];
    }

    flat_data.resize(total_elements);
    int idx = 0;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < row_sizes[i]; j++) {
        flat_data[idx++] = matrix[i][j];
      }
    }
  }

  std::vector<int> counts(size, 0);
  std::vector<int> displacements(size, 0);

  int total_rows_per_process = rows / size;
  int remainder = rows % size;

  for (int i = 0; i < size; i++) {
    counts[i] = total_rows_per_process + (i < remainder ? 1 : 0);
    if (i > 0) {
      displacements[i] = displacements[i - 1] + counts[i - 1];
    }
  }

  std::vector<int> sendcounts(size, 0);
  std::vector<int> senddispls(size, 0);

  if (rank == 0) {
    for (int i = 0; i < size; i++) {
      senddispls[i] = (i == 0) ? 0 : senddispls[i - 1] + sendcounts[i - 1];
      for (int j = 0; j < counts[i]; j++) {
        int row_idx = displacements[i] + j;
        sendcounts[i] += row_sizes[row_idx];
      }
    }
  }

  int my_element_count = 0;
  MPI_Scatter(sendcounts.data(), 1, MPI_INT, &my_element_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_data(my_element_count);
  MPI_Scatterv(flat_data.data(), sendcounts.data(), senddispls.data(), MPI_INT, local_data.data(), my_element_count,
               MPI_INT, 0, MPI_COMM_WORLD);

  int64_t local_sum = 0;
  for (int val : local_data) {
    local_sum += val;
  }

  int64_t global_sum = 0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_sum;
  }

  return true;
}

bool FrolovaSSumElemMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace frolova_s_sum_elem_matrix
