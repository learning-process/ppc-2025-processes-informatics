#include "rozenberg_a_matrix_column_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "rozenberg_a_matrix_column_sum/common/include/common.hpp"

namespace rozenberg_a_matrix_column_sum {

RozenbergAMatrixColumnSumMPI::RozenbergAMatrixColumnSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  InType empty;
  GetInput().swap(empty);

  for (const auto &row : in) {
    GetInput().push_back(row);
  }

  GetOutput().clear();
}

bool RozenbergAMatrixColumnSumMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return (!(GetInput().empty())) && (GetOutput().empty());
  }
  return true;
}

bool RozenbergAMatrixColumnSumMPI::PreProcessingImpl() {
  return true;
}

bool RozenbergAMatrixColumnSumMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows;
  int columns;
  if (rank == 0) {
    rows = static_cast<int>(GetInput().size());
    columns = static_cast<int>(GetInput()[0].size());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int chunk = rows / size;
  int remainder = rows % size;

  int rows_count = chunk + (rank < remainder ? 1 : 0);

  std::vector<int> sendcounts;
  std::vector<int> displs;
  std::vector<int> flat;

  if (rank == 0) {
    flat.resize(rows * columns);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        flat[j + i * columns] = GetInput()[i][j];
      }
    }

    sendcounts.resize(size);
    displs.resize(size);

    int offset = 0;
    for (int p = 0; p < size; p++) {
      int r = chunk + (p < remainder ? 1 : 0);
      sendcounts[p] = r * columns;
      displs[p] = offset;
      offset += r * columns;
    }
  }

  std::vector<int> local_buf(rows_count * columns);

  MPI_Scatterv(rank == 0 ? flat.data() : nullptr, rank == 0 ? sendcounts.data() : nullptr,
               rank == 0 ? displs.data() : nullptr, MPI_INT, local_buf.data(), rows_count * columns, MPI_INT, 0,
               MPI_COMM_WORLD);

  OutType local_res(columns, 0);
  for (int i = 0; i < rows_count; i++) {
    for (int j = 0; j < columns; j++) {
      local_res[j] += local_buf[j + i * columns];
    }
  }

  if (rank == 0) {
    GetOutput().resize(GetInput()[0].size());
  }

  MPI_Reduce(local_res.data(), rank == 0 ? GetOutput().data() : nullptr, columns, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  return true;
}

bool RozenbergAMatrixColumnSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rozenberg_a_matrix_column_sum
