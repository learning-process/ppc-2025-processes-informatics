#include "sannikov_i_column_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "sannikov_i_column_sum/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sannikov_i_column_sum {

SannikovIColumnSumMPI::SannikovIColumnSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &dst = GetInput();
  InType tmp(in);
  dst.swap(tmp);

  GetOutput().clear();
}

bool SannikovIColumnSumMPI::ValidationImpl() {
  const auto &input_matrix = GetInput();
  return (!input_matrix.empty()) && (input_matrix.front().size() != 0) && (GetOutput().empty());
}

bool SannikovIColumnSumMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const auto &input_matrix = GetInput();
  int columns = 0;
  if (rank == 0) {
    if (input_matrix.empty()) {
      return false;
    }
    columns = (input_matrix.front().size());
  }
  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (columns <= 0) {
    return false;
  }
  GetOutput().resize((columns), 0);
  return !GetOutput().empty();
}

bool SannikovIColumnSumMPI::RunImpl() {
  const auto &input_matrix = GetInput();

  int rank = 0;
  int size = 1;
  int rows = 0;
  int columns = (GetOutput().size());

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0) {
    if (input_matrix.empty() || (int)input_matrix.front().size() != columns) {
      return false;
    }
    rows = (input_matrix.size());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rows <= 0 || columns <= 0) {
    return false;
  }
  int rem = rows % size;
  int pad = (rem == 0) ? 0 : (size - rem);
  int total_rows = rows + pad;
  int rows_per_proc = total_rows / size;
  std::vector<int> sendbuf;
  if (rank == 0) {
    sendbuf.resize((total_rows * columns));
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        sendbuf[(i)*columns + j] = input_matrix[(i)][(j)];
      }
    }

    for (int i = rows; i < total_rows; i++) {
      for (int j = 0; j < columns; j++) {
        sendbuf[(i)*columns + j] = 0;
      }
    }
  }

  std::vector<int> recvbuf((rows_per_proc * columns), 0);
  MPI_Scatter(rank == 0 ? sendbuf.data() : nullptr, rows_per_proc * columns, MPI_INT, recvbuf.data(),
              rows_per_proc * columns, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<int> sum((columns), 0);
  for (int i = 0; i < rows_per_proc; i++) {
    int local_id = i * columns;
    for (int j = 0; j < columns; j++) {
      sum[(j)] += recvbuf[local_id + j];
    }
  }
  MPI_Allreduce(sum.data(), GetOutput().data(), columns, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return !GetOutput().empty();
}

bool SannikovIColumnSumMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace sannikov_i_column_sum
