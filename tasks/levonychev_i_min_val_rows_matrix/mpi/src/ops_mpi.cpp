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
  std::cout << "ROWS = " << ROWS << ", COLS = " << COLS << std::endl;
  OutType &global_min_values = GetOutput();
  if (global_min_values.size() != static_cast<size_t>(ROWS)) {
    global_min_values.resize(ROWS);
  }
  int ProcNum, ProcRank;
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  MPI_Comm new_comm;
  MPI_Comm_split(MPI_COMM_WORLD, ROWS * (COLS + 1), ProcRank, &new_comm);
  MPI_Comm_size(new_comm, &ProcNum);
  MPI_Comm_rank(new_comm, &ProcRank);

  int local_count_of_rows;
  int start_id;
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

  std::vector<double> local_min_values(local_count_of_rows);
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
  std::cout << ProcRank << "/" << ProcNum << "__" << local_count_of_rows << ": ";
  for (auto j : local_min_values) {
    std::cout << j << ' ';
  }
  std::cout << std::endl;
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
  MPI_Gatherv(local_min_values.data(), local_count_of_rows, MPI_DOUBLE, global_min_values.data(), recvcounts.data(),
              displs.data(), MPI_DOUBLE, 0, new_comm);
  MPI_Bcast(global_min_values.data(), ROWS, MPI_DOUBLE, 0, new_comm);
  std::cout << ProcRank << ": ";
  for (auto i : global_min_values) {
    std::cout << i << ' ';
  }
  std::cout << std::endl;
  MPI_Comm_free(&new_comm);
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::PostProcessingImpl() {
  return GetOutput().size() == std::get<1>(GetInput());
}

}  // namespace levonychev_i_min_val_rows_matrix
