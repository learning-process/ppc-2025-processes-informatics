#include "dorofeev_i_ccs_martrix_production/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "dorofeev_i_ccs_martrix_production/common/include/common.hpp"

namespace dorofeev_i_ccs_martrix_production {

DorofeevICCSMatrixProductionMPI::DorofeevICCSMatrixProductionMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool DorofeevICCSMatrixProductionMPI::ValidationImpl() {
  const auto &a = GetInput().first;
  const auto &b = GetInput().second;

  if (a.cols != b.rows) {
    return false;
  }

  if (static_cast<int>(a.col_ptr.size()) != a.cols + 1) {
    return false;
  }

  if (static_cast<int>(b.col_ptr.size()) != b.cols + 1) {
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

bool DorofeevICCSMatrixProductionMPI::RunImpl() {  // NOLINT(readability-function-cognitive-complexity)
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &a = GetInput().first;
  const auto &b = GetInput().second;
  auto &c = GetOutput();

  c.rows = a.rows;
  c.cols = b.cols;
  c.col_ptr.assign(b.cols + 1, 0);
  c.row_indices.clear();
  c.values.clear();

  const int cols_per_proc = b.cols / size;
  const int remainder = b.cols % size;

  const int start_col = (rank * cols_per_proc) + std::min(rank, remainder);
  const int local_cols = cols_per_proc + (rank < remainder ? 1 : 0);

  std::vector<std::map<int, double>> local_columns(local_cols);

  for (int j = 0; j < local_cols; j++) {
    int global_col = start_col + j;
    std::map<int, double> col_map;
    int start = b.col_ptr[global_col];
    int end = b.col_ptr[global_col + 1];
    for (int idx = start; idx < end; ++idx) {
      int k = b.row_indices[idx];
      double b_val = b.values[idx];
      int a_start = a.col_ptr[k];
      int a_end = a.col_ptr[k + 1];
      for (int a_idx = a_start; a_idx < a_end; ++a_idx) {
        int m = a.row_indices[a_idx];
        double a_val = a.values[a_idx];
        col_map[m] += a_val * b_val;
      }
    }
    local_columns[j] = col_map;
  }

  if (rank == 0) {
    int col_offset = 0;
    for (int j = 0; j < local_cols; j++) {
      c.col_ptr[col_offset + 1] = c.col_ptr[col_offset] + static_cast<int>(local_columns[j].size());
      for (const auto &[row, value] : local_columns[j]) {
        c.row_indices.push_back(row);
        c.values.push_back(value);
      }
      col_offset++;
    }

    for (int process_idx = 0; process_idx < size - 1; process_idx++) {
      int recv_cols = cols_per_proc + (process_idx + 1 < remainder ? 1 : 0);
      for (int j = 0; j < recv_cols; j++) {
        int nnz = 0;
        MPI_Recv(&nnz, 1, MPI_INT, process_idx + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        c.col_ptr[col_offset + 1] = c.col_ptr[col_offset] + nnz;
        for (int k = 0; k < nnz; k++) {
          int row = 0;
          double val = 0.0;
          MPI_Recv(&row, 1, MPI_INT, process_idx + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(&val, 1, MPI_DOUBLE, process_idx + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          c.row_indices.push_back(row);
          c.values.push_back(val);
        }
        col_offset++;
      }
    }
  } else {
    for (const auto &col : local_columns) {
      int nnz = static_cast<int>(col.size());
      MPI_Send(&nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      for (const auto &[row, val] : col) {
        MPI_Send(&row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&val, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Broadcast the result to all processes
  MPI_Bcast(&c.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&c.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int col_ptr_size = static_cast<int>(c.col_ptr.size());
  MPI_Bcast(&col_ptr_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    c.col_ptr.resize(col_ptr_size);
  }
  MPI_Bcast(c.col_ptr.data(), col_ptr_size, MPI_INT, 0, MPI_COMM_WORLD);
  int row_indices_size = static_cast<int>(c.row_indices.size());
  MPI_Bcast(&row_indices_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    c.row_indices.resize(row_indices_size);
  }
  MPI_Bcast(c.row_indices.data(), row_indices_size, MPI_INT, 0, MPI_COMM_WORLD);
  int values_size = static_cast<int>(c.values.size());
  MPI_Bcast(&values_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    c.values.resize(values_size);
  }
  MPI_Bcast(c.values.data(), values_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool DorofeevICCSMatrixProductionMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dorofeev_i_ccs_martrix_production
