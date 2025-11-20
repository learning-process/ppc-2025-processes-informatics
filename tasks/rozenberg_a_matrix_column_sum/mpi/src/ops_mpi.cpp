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
  return (!(GetInput().empty())) && (GetOutput().empty());
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

  int rows = static_cast<int>(GetInput().size());
  int columns = static_cast<int>(GetInput()[0].size());

  int chunk = rows / size;
  int remainder = rows % size;

  int begin = (chunk * rank) + std::min(rank, remainder);
  int end = begin + chunk + (rank < remainder ? 1 : 0);

  OutType local_res(columns, 0);
  for (int i = begin; i < end; i++) {
    for (int j = 0; j < columns; j++) {
      local_res[j] += GetInput()[i][j];
    }
  }

  MPI_Allreduce(local_res.data(), GetOutput().data(), columns, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  return true;
}

bool RozenbergAMatrixColumnSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rozenberg_a_matrix_column_sum
