#include "levonychev_i_min_val_rows_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "levonychev_i_min_val_rows_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_min_val_rows_matrix {

LevonychevIMinValRowsMatrixMPI::LevonychevIMinValRowsMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(std::get<1>(in));
}
bool LevonychevIMinValRowsMatrixMPI::ValidationImpl() {
  if (std::get<0>(GetInput()).size() == 0 || std::get<1>(GetInput()) == 0 || std::get<2>(GetInput()) == 0) {
    return false;
  }
  if (std::get<0>(GetInput()).size() != std::get<1>(GetInput()) * std::get<2>(GetInput())) {
    return false;
  }
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::PreProcessingImpl() {
  GetOutput().resize(std::get<1>(GetInput()));
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::RunImpl() {
  const std::vector<double> &matrix = std::get<0>(GetInput());
  const int ROWS = std::get<1>(GetInput());
  const int COLS = std::get<2>(GetInput());
  OutType &global_min_values = GetOutput();
  int ProcNum, ProcRank;
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

  int local_count_of_rows = ROWS / ProcNum;
  if ((ROWS % ProcNum != 0) && (ProcRank == ProcNum - 1)) {
    local_count_of_rows += (ROWS % ProcNum);
  }

  std::vector<double> local_min_values(local_count_of_rows);
  const int start_id = ProcRank * (ROWS / ProcNum) * COLS;
  for (int i = 0; i < local_count_of_rows; ++i) {
    const int start_row_id = start_id + COLS * i;
    double min_value = matrix[start_row_id];
    for (int j = 1; j < COLS; ++j) {
      if (matrix[start_row_id + j] < min_value) {
        min_value = matrix[start_row_id + j];
      }
    }
    local_min_values[i] = min_value;
  }
  std::cout << ProcRank << "__" << local_count_of_rows << ": ";
  for (auto j : local_min_values) {
    std::cout << j << ' ';
  }
  std::cout << std::endl;

  if (ProcRank == 0) {
    for (int i = 0; i < local_count_of_rows; ++i) {
      global_min_values[i] = local_min_values[i];
    }
  }

  bool send = false;
  for (int i = 1; i < ProcNum; ++i) {
    if (!send && ProcRank != 0) {
      MPI_Send(&local_count_of_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      MPI_Send(local_min_values.data(), local_count_of_rows, MPI_DOUBLE, 0, ProcRank, MPI_COMM_WORLD);
      send = true;
    }
    if (ProcRank == 0) {
      std::vector<double> other_local_min_values;
      MPI_Status status_buf;
      MPI_Status status_size;
      int count;
      MPI_Recv(&count, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status_size);
      other_local_min_values.resize(count);
      MPI_Recv(other_local_min_values.data(), count, MPI_DOUBLE, status_size.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
               &status_buf);
      int proc_rank_source = status_buf.MPI_SOURCE;
      for (int j = 0; j < count; ++j) {
        global_min_values[j + (ROWS / ProcNum) * proc_rank_source] = other_local_min_values[j];
      }
    }
  }
  MPI_Bcast(global_min_values.data(), global_min_values.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  std::cout << ProcRank << ": ";
  for (auto i : global_min_values) {
    std::cout << i << ' ';
  }
  std::cout << std::endl;
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::PostProcessingImpl() {
  return GetOutput().size() == std::get<1>(GetInput());
}

}  // namespace levonychev_i_min_val_rows_matrix
