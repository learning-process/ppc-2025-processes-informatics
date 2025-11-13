#include "sannikov_i_column_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "sannikov_i_column_sum/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sannikov_i_column_sum {

SannikovIColumnSumMPI::SannikovIColumnSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SannikovIColumnSumMPI::ValidationImpl() {
  return (!GetInput().empty()) && (GetInput().front().size() != 0) && (GetOutput().empty());
}

bool SannikovIColumnSumMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int columns = 0;
  if (rank == 0) {
    if (GetInput().empty()) {
      return false;
    }
    columns = (GetInput().front().size());
  }
  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (columns <= 0) {
    return false;
  }
  GetOutput().resize((columns), 0);
  return !GetOutput().empty();
}

bool SannikovIColumnSumMPI::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }
  int rank = 0;
  int size = 1;
  size_t rows = 0;
  size_t columns = (GetOutput().size());

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0) {
    if (GetInput().empty() || (size_t)GetInput().front().size() != columns) {
      return false;
    }
    rows = (GetInput().size());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rows <= 0 || columns <= 0) {
    return false;
  }
  size_t rem = rows % size;
  size_t pad = (rem == 0) ? 0 : (size - rem);
  size_t total_rows = rows + pad;
  size_t rows_per_proc = total_rows / size;
  std::vector<int> sendbuf;
  if (rank == 0) {
    sendbuf.resize((total_rows * columns));
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < columns; j++) {
        sendbuf[(i)*columns + j] = GetInput()[(i)][(j)];
      }
    }

    for (size_t i = rows; i < total_rows; i++) {
      for (size_t j = 0; j < columns; j++) {
        sendbuf[(i)*columns + j] = 0;
      }
    }
  }

  std::vector<int> recvbuf((rows_per_proc * columns), 0);
  MPI_Scatter(rank == 0 ? sendbuf.data() : nullptr, rows_per_proc * columns, MPI_INT, recvbuf.data(),
              rows_per_proc * columns, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<int> sum((columns), 0);
  for (size_t i = 0; i < rows_per_proc; i++) {
    int local_id = i * columns;
    for (size_t j = 0; j < columns; j++) {
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
