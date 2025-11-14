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
  GetOutput() = {};
}
bool LevonychevIMinValRowsMatrixMPI::ValidationImpl() {
  const size_t vector_size_ = std::get<0>(GetInput()).size();
  const int ROWS = std::get<1>(GetInput());
  const int COLS = std::get<2>(GetInput());
  if (vector_size_ == 0 || ROWS == 0 || COLS == 0) {
    return false;
  }
  if (vector_size_ != static_cast<size_t>(ROWS * COLS)) {
    return false;
  }
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::PreProcessingImpl() {
  GetOutput().resize(std::get<1>(GetInput()));
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::RunImpl() {
  const std::vector<int> &matrix = std::get<0>(GetInput());
  const int ROWS = std::get<1>(GetInput());
  const int COLS = std::get<2>(GetInput());
  OutType &global_min_values = GetOutput();

  if (global_min_values.size() != static_cast<size_t>(ROWS)) {
    global_min_values.resize(ROWS);
  }
  int ProcNum = 0;
  int ProcRank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

  int local_count_of_rows = 0;
  int start_id = 0;
  if (ROWS < ProcNum) {
    if (ProcRank < ROWS) {
      local_count_of_rows = 1;
      start_id = ProcRank * COLS;
    } else {
      local_count_of_rows = 0;
      start_id = 0;
    }
  } else {
    local_count_of_rows = ROWS / ProcNum;
    if (ProcRank == (ProcNum - 1)) {
      local_count_of_rows += (ROWS % ProcNum);
    }
    start_id = ProcRank * (ROWS / ProcNum) * COLS;
  }

  std::vector<int> local_min_values(local_count_of_rows);
  for (int i = 0; i < local_count_of_rows; ++i) {
    const int start_row_id = start_id + COLS * i;
    int min_value = matrix[start_row_id];
    for (int j = 1; j < COLS; ++j) {
      if (matrix[start_row_id + j] < min_value) {
        min_value = matrix[start_row_id + j];
      }
    }
    local_min_values[i] = min_value;
  }

  std::vector<int> recvcounts(ProcNum);
  std::vector<int> displs(ProcNum);
  int current_displacement = 0;

  for (int i = 0; i < ProcNum; ++i) {
    int count_i;
    if (ROWS < ProcNum) {
      if (i < ROWS) {
        count_i = 1;
      } else {
        count_i = 0;
      }
    } else {
      count_i = ROWS / ProcNum;
      if (i == (ProcNum - 1)) {
        count_i += ROWS % ProcNum;
      }
    }
    recvcounts[i] = count_i;

    displs[i] = current_displacement;
    current_displacement += count_i;
  }
  MPI_Gatherv(local_min_values.data(), local_count_of_rows, MPI_INT, global_min_values.data(), recvcounts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(global_min_values.data(), ROWS, MPI_INT, 0, MPI_COMM_WORLD);
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::PostProcessingImpl() {
  return GetOutput().size() == std::get<1>(GetInput());
}

}  // namespace levonychev_i_min_val_rows_matrix
