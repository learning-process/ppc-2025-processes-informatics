#include <mpi.h>
#include <algorithm>
#include <vector>
#include "makovskiy_i_min_value_in_matrix_rows/mpi/include/ops_mpi.hpp"

namespace makovskiy_i_min_value_in_matrix_rows {

MinValueMPI::MinValueMPI(const InType &in) {
  this->GetInput() = in;
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool MinValueMPI::ValidationImpl() {
  const auto &mat = this->GetInput();
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    if (mat.empty()) {
      return false;
    }
    for (const auto &row : mat) {
      if (row.empty()) {
        return false;
      }
    }
  }
  return true;
}

bool MinValueMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto &mat = this->GetInput();
    this->GetOutput().clear();
    this->GetOutput().resize(mat.size());
  }
  return true;
}

bool MinValueMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> local_min_values;

  if (rank == 0) {
    const auto &matrix = this->GetInput();
    int num_rows = matrix.size();
    int rows_per_proc = num_rows / size;
    int remaining_rows = num_rows % size;

    int current_row_idx = 0;
    for (int i = 0; i < size; ++i) {
      int rows_for_this_proc = rows_per_proc + (i < remaining_rows ? 1 : 0);
      if (i == 0) {
        for (int j = 0; j < rows_for_this_proc; ++j) {
          const auto &row = matrix[current_row_idx++];
          local_min_values.push_back(*std::min_element(row.begin(), row.end()));
        }
      } else {
        MPI_Send(&rows_for_this_proc, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        for (int j = 0; j < rows_for_this_proc; ++j) {
          const auto &row = matrix[current_row_idx++];
          int row_size = row.size();
          MPI_Send(&row_size, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
          MPI_Send(row.data(), row_size, MPI_INT, i, 2, MPI_COMM_WORLD);
        }
      }
    }
  } else {
    int num_local_rows;
    MPI_Recv(&num_local_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < num_local_rows; ++i) {
      int row_size;
      MPI_Recv(&row_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::vector<int> received_row(row_size);
      MPI_Recv(received_row.data(), row_size, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_min_values.push_back(*std::min_element(received_row.begin(), received_row.end()));
    }
  }

  int local_results_count = local_min_values.size();
  std::vector<int> recvcounts;
  if (rank == 0) {
    recvcounts.resize(size);
  }

  MPI_Gather(&local_results_count, 1, MPI_INT, (rank == 0 ? recvcounts.data() : nullptr), 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  std::vector<int> displs;
  if (rank == 0) {
    displs.resize(size);
    if (!displs.empty()) {
      displs[0] = 0;
    }
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
  }

  MPI_Gatherv(local_min_values.data(), local_results_count, MPI_INT, (rank == 0 ? this->GetOutput().data() : nullptr),
              (rank == 0 ? recvcounts.data() : nullptr), (rank == 0 ? displs.data() : nullptr), MPI_INT, 0,
              MPI_COMM_WORLD);

  return true;
}

bool MinValueMPI::PostProcessingImpl() {
  return true;
}

}