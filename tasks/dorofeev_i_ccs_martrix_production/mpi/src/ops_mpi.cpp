#include "dorofeev_i_ccs_martrix_production/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <map>
#include <vector>

#include "dorofeev_i_ccs_martrix_production/common/include/common.hpp"

namespace dorofeev_i_ccs_martrix_production {

DorofeevICCSMatrixProductionMPI::DorofeevICCSMatrixProductionMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool DorofeevICCSMatrixProductionMPI::ValidationImpl() {
  const auto &A = GetInput().first;
  const auto &B = GetInput().second;

  if (A.cols != B.rows) {
    return false;
  }

  if (A.col_ptr.size() != static_cast<size_t>(A.cols + 1)) {
    return false;
  }

  if (B.col_ptr.size() != static_cast<size_t>(B.cols + 1)) {
    return false;
  }

  return true;
}

bool DorofeevICCSMatrixProductionMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput().rows = GetInput().first.rows;
    GetOutput().cols = GetInput().second.cols;
    GetOutput().col_ptr.assign(GetOutput().cols + 1, 0);
  }

  return true;
}

bool DorofeevICCSMatrixProductionMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &A = GetInput().first;
  const auto &B = GetInput().second;
  auto &C = GetOutput();

  C.rows = A.rows;
  C.cols = B.cols;
  C.col_ptr.assign(B.cols + 1, 0);
  C.row_indices.clear();
  C.values.clear();

  const int cols_per_proc = B.cols / size;
  const int remainder = B.cols % size;

  const int start_col = rank * cols_per_proc + std::min(rank, remainder);
  const int local_cols = cols_per_proc + (rank < remainder ? 1 : 0);

  std::vector<std::map<int, double>> local_columns(local_cols);

  for (int j = 0; j < local_cols; j++) {
    int global_col = start_col + j;
    std::map<int, double> col_map;
    int start = B.col_ptr[global_col];
    int end = B.col_ptr[global_col + 1];
    for (int idx = start; idx < end; ++idx) {
      int k = B.row_indices[idx];
      double b_val = B.values[idx];
      int a_start = A.col_ptr[k];
      int a_end = A.col_ptr[k + 1];
      for (int a_idx = a_start; a_idx < a_end; ++a_idx) {
        int m = A.row_indices[a_idx];
        double a_val = A.values[a_idx];
        col_map[m] += a_val * b_val;
      }
    }
    local_columns[j] = col_map;
  }

  if (rank == 0) {
    int col_offset = 0;
    for (int j = 0; j < local_cols; j++) {
      C.col_ptr[col_offset + 1] = C.col_ptr[col_offset] + local_columns[j].size();
      for (const auto &[row, value] : local_columns[j]) {
        C.row_indices.push_back(row);
        C.values.push_back(value);
      }
      col_offset++;
    }

    for (int p = 0; p < size - 1; p++) {
      int recv_cols = cols_per_proc + (p + 1 < remainder ? 1 : 0);
      for (int j = 0; j < recv_cols; j++) {
        int nnz = 0;
        MPI_Recv(&nnz, 1, MPI_INT, p + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        C.col_ptr[col_offset + 1] = C.col_ptr[col_offset] + nnz;
        for (int k = 0; k < nnz; k++) {
          int row;
          double val;
          MPI_Recv(&row, 1, MPI_INT, p + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(&val, 1, MPI_DOUBLE, p + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          C.row_indices.push_back(row);
          C.values.push_back(val);
        }
        col_offset++;
      }
    }
  } else {
    for (const auto &col : local_columns) {
      int nnz = col.size();
      MPI_Send(&nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      for (const auto &[row, val] : col) {
        MPI_Send(&row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&val, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Broadcast the result to all processes
  MPI_Bcast(&C.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&C.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int col_ptr_size = C.col_ptr.size();
  MPI_Bcast(&col_ptr_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    C.col_ptr.resize(col_ptr_size);
  }
  MPI_Bcast(C.col_ptr.data(), col_ptr_size, MPI_INT, 0, MPI_COMM_WORLD);
  int row_indices_size = C.row_indices.size();
  MPI_Bcast(&row_indices_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    C.row_indices.resize(row_indices_size);
  }
  MPI_Bcast(C.row_indices.data(), row_indices_size, MPI_INT, 0, MPI_COMM_WORLD);
  int values_size = C.values.size();
  MPI_Bcast(&values_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    C.values.resize(values_size);
  }
  MPI_Bcast(C.values.data(), values_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool DorofeevICCSMatrixProductionMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dorofeev_i_ccs_martrix_production
