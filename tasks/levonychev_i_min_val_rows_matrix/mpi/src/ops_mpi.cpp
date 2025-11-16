#include "levonychev_i_min_val_rows_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "levonychev_i_min_val_rows_matrix/common/include/common.hpp"

namespace levonychev_i_min_val_rows_matrix {

LevonychevIMinValRowsMatrixMPI::LevonychevIMinValRowsMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}
bool LevonychevIMinValRowsMatrixMPI::ValidationImpl() {
  const size_t vector_size = std::get<0>(GetInput()).size();
  const int rows = std::get<1>(GetInput());
  const int cols = std::get<2>(GetInput());
  return vector_size != 0 && rows != 0 && cols != 0 &&
         (vector_size == static_cast<size_t>(rows) * static_cast<size_t>(cols));
}

bool LevonychevIMinValRowsMatrixMPI::PreProcessingImpl() {
  GetOutput().resize(std::get<1>(GetInput()));
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::RunImpl() {
  const std::vector<int> &matrix = std::get<0>(GetInput());
  const int rows = std::get<1>(GetInput());
  const int cols = std::get<2>(GetInput());
  OutType &global_min_values = GetOutput();

  int proc_num = 0;
  int proc_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  int local_count_of_rows = rows / proc_num;
  if (proc_rank == (proc_num - 1)) {
    local_count_of_rows += (rows % proc_num);
  }
  int start_id = proc_rank * (rows / proc_num) * cols;

  std::vector<int> local_min_values(local_count_of_rows);
  for (int i = 0; i < local_count_of_rows; ++i) {
    const int start_row_id = start_id + (cols * i);
    int min_value = matrix[start_row_id];
    for (int j = 1; j < cols; ++j) {
      min_value = std::min(matrix[start_row_id + j], min_value);
    }
    local_min_values[i] = min_value;
  }

  std::vector<int> recvcounts(proc_num);
  std::vector<int> displs(proc_num);
  int current_displacement = 0;

  for (int i = 0; i < proc_num; ++i) {
    int count_i = rows / proc_num;
    if (i == (proc_num - 1)) {
      count_i += rows % proc_num;
    }
    recvcounts[i] = count_i;
    displs[i] = current_displacement;
    current_displacement += count_i;
  }
  MPI_Gatherv(local_min_values.data(), local_count_of_rows, MPI_INT, global_min_values.data(), recvcounts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(global_min_values.data(), rows, MPI_INT, 0, MPI_COMM_WORLD);
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace levonychev_i_min_val_rows_matrix
