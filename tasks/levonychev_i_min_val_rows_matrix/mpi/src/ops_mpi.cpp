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
  std::cout << "ROWS = " << ROWS << ", COLS = " << COLS << std::endl;
  OutType &global_min_values = GetOutput();
  int ProcNum, ProcRank;
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  int local_count_of_rows = ROWS / ProcNum;
  if (ProcRank == (ProcNum - 1)) {
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
  std::cout << ProcRank << "/" << ProcNum << "__" << local_count_of_rows << ": ";
  for (auto j : local_min_values) {
    std::cout << j << ' ';
  }
  std::cout << std::endl;
  std::vector<int> recvcounts(ProcNum);
  std::vector<int> displs(ProcNum);
  for (int i = 0; i < ProcNum; ++i) {
    recvcounts[i] = ROWS / ProcNum;
  }
  recvcounts[recvcounts.size() - 1] += ROWS % ProcNum;

  for (int i = 0; i < ProcNum; ++i) {
    displs[i] = (ROWS / ProcNum) * i;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Gatherv(local_min_values.data(), local_count_of_rows, MPI_DOUBLE, global_min_values.data(), recvcounts.data(),
              displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
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
