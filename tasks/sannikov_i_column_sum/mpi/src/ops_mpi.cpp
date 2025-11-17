#include "sannikov_i_column_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "sannikov_i_column_sum/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sannikov_i_column_sum {

SannikovIColumnSumMPI::SannikovIColumnSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &input_buffer = GetInput();
  InType tmp(in);
  input_buffer.swap(tmp);
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
    if (input_matrix.empty() || static_cast<int>(input_matrix.front().size()) != columns) {
      return false;
    }
    rows = (input_matrix.size());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rows <= 0 || columns <= 0) {
    return false;
  }
  std::vector<int> sendbuf;
  if (rank == 0) {
    sendbuf.resize((rows * columns));
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        sendbuf[i * columns + j] = input_matrix[i][j];
      }
    }
  }
  std::vector<int> elem_for_proc(size);
  std::vector<int> id_elem(size);
  int displacement = 0;
  for (int i = 0; i < size; i++) {
    elem_for_proc[i] = rows * columns / size + (i < rows * columns % size ? 1 : 0);
    id_elem[i] = displacement;
    displacement += elem_for_proc[i];
  }
  int mpi_displacement = 0;
  mpi_displacement = id_elem[rank] % columns;
  std::vector<int> buf(elem_for_proc[rank], 0);
  MPI_Scatterv(rank == 0 ? sendbuf.data() : nullptr, elem_for_proc.data(), id_elem.data(), MPI_INT, buf.data(),
               elem_for_proc[rank], MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<int> sum(columns, 0);
  for (int i = 0; i < elem_for_proc[rank]; i++) {
    int new_col = (i + mpi_displacement) % columns;
    sum[new_col] += buf[i];
  }
  MPI_Allreduce(sum.data(), GetOutput().data(), columns, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return !GetOutput().empty();
}

bool SannikovIColumnSumMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace sannikov_i_column_sum
