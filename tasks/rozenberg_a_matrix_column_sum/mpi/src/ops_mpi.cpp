#include "rozenberg_a_matrix_column_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "rozenberg_a_matrix_column_sum/common/include/common.hpp"

namespace rozenberg_a_matrix_column_sum {

RozenbergAMatrixColumnSumMPI::RozenbergAMatrixColumnSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool RozenbergAMatrixColumnSumMPI::ValidationImpl() {
  bool rows_empty = false;
  for (const auto &i : GetInput()) {
    if (i.empty()) {
      rows_empty = true;
      break;
    }
  }
  return (!(GetInput().empty())) && (GetOutput().empty()) && (!rows_empty);
}

bool RozenbergAMatrixColumnSumMPI::PreProcessingImpl() {
  GetOutput().resize(GetInput()[0].size());
  return GetOutput().size() == GetInput()[0].size();
}

bool RozenbergAMatrixColumnSumMPI::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t rows = GetInput().size();
  size_t columns = GetInput()[0].size();

  int chunk = (int)rows / size;
  int remainder = (int)rows % size;

  int begin = (chunk * rank) + std::min(rank, remainder);
  int end = begin + chunk + (rank < remainder ? 1 : 0);

  OutType local_res(columns, 0);
  for (int i = begin; i < end; i++) {
    for (int j = 0; j < columns; j++) {
      local_res[j] += GetInput()[i][j];
    }
  }

  MPI_Allreduce(local_res.data(), GetOutput().data(), columns, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  return !(GetOutput().empty());
}

bool RozenbergAMatrixColumnSumMPI::PostProcessingImpl() {
  return !(GetOutput().empty());
}

}  // namespace rozenberg_a_matrix_column_sum
