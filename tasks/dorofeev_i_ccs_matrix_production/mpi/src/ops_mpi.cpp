#include "dorofeev_i_ccs_matrix_production/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dorofeev_i_ccs_matrix_production/common/include/common.hpp"

namespace dorofeev_i_ccs_matrix_production {

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

std::pair<int, int> DorofeevICCSMatrixProductionMPI::ComputeLocalColumnRange(int rank, int size, int total_cols) {
  const int cols_per_proc = total_cols / size;
  const int remainder = total_cols % size;
  const int start_col = (rank * cols_per_proc) + std::min(rank, remainder);
  const int local_cols = cols_per_proc + (rank < remainder ? 1 : 0);
  return {start_col, local_cols};
}

std::vector<std::unordered_map<int, double>> DorofeevICCSMatrixProductionMPI::ComputeLocalColumns(const InType &input,
                                                                                                  int start_col,
                                                                                                  int local_cols) {
  const auto &a = input.first;
  const auto &b = input.second;
  std::vector<std::unordered_map<int, double>> local_columns(local_cols);

  for (size_t j = 0; std::cmp_less(j, local_cols); j++) {
    int global_col = start_col + static_cast<int>(j);
    std::unordered_map<int, double> col_map;
    int start = b.col_ptr[global_col];
    int end = b.col_ptr[global_col + 1];
    auto row_it = b.row_indices.begin() + start;
    auto val_it = b.values.begin() + start;
    for (; row_it != b.row_indices.begin() + end; ++row_it, ++val_it) {
      int k = *row_it;
      double b_val = *val_it;
      int a_start = a.col_ptr[k];
      int a_end = a.col_ptr[k + 1];
      auto a_row_it = a.row_indices.begin() + a_start;
      auto a_val_it = a.values.begin() + a_start;
      for (; a_row_it != a.row_indices.begin() + a_end; ++a_row_it, ++a_val_it) {
        int m = *a_row_it;
        double a_val = *a_val_it;
        col_map[m] += a_val * b_val;
      }
    }
    local_columns[j] = col_map;
  }

  return local_columns;
}

void DorofeevICCSMatrixProductionMPI::GatherResultsOnRoot(OutType &output,
                                                          std::vector<std::unordered_map<int, double>> &local_columns,
                                                          int size, int rank) {
  if (rank == 0) {
    int col_offset = 0;
    ProcessLocalResultsOnRoot(output, local_columns, col_offset);
    for (int process_idx = 1; process_idx < size; process_idx++) {
      ReceiveResultsFromProcess(output, process_idx, col_offset);
    }
  } else {
    SendResultsToRoot(local_columns);
  }
}

void DorofeevICCSMatrixProductionMPI::ProcessLocalResultsOnRoot(
    OutType &output, std::vector<std::unordered_map<int, double>> &local_columns, int &col_offset) {
  for (auto &local_column : local_columns) {
    std::vector<std::pair<int, double>> sorted_col(local_column.begin(), local_column.end());
    std::ranges::sort(sorted_col);
    output.col_ptr[col_offset + 1] = output.col_ptr[col_offset] + static_cast<int>(sorted_col.size());
    for (const auto &[row, value] : sorted_col) {
      output.row_indices.push_back(row);
      output.values.push_back(value);
    }
    col_offset++;
  }
}

void DorofeevICCSMatrixProductionMPI::ReceiveResultsFromProcess(OutType &output, int process_idx, int &col_offset) {
  int num_cols = 0;
  MPI_Recv(&num_cols, 1, MPI_INT, process_idx, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  std::vector<int> nnzs(num_cols);
  MPI_Recv(nnzs.data(), num_cols, MPI_INT, process_idx, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  int total_nnz = 0;
  for (int n : nnzs) {
    total_nnz += n;
  }
  std::vector<int> all_rows(total_nnz);
  MPI_Recv(all_rows.data(), total_nnz, MPI_INT, process_idx, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  std::vector<double> all_vals(total_nnz);
  MPI_Recv(all_vals.data(), total_nnz, MPI_DOUBLE, process_idx, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  auto row_it = all_rows.begin();
  auto val_it = all_vals.begin();
  for (int n : nnzs) {
    output.col_ptr[col_offset + 1] = output.col_ptr[col_offset] + n;
    for (int k = 0; k < n; k++) {
      output.row_indices.push_back(*row_it);
      output.values.push_back(*val_it);
      ++row_it;
      ++val_it;
    }
    col_offset++;
  }
}

void DorofeevICCSMatrixProductionMPI::SendResultsToRoot(
    const std::vector<std::unordered_map<int, double>> &local_columns) {
  int num_cols = static_cast<int>(local_columns.size());
  MPI_Send(&num_cols, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  std::vector<int> nnzs;
  std::vector<int> all_rows;
  std::vector<double> all_vals;
  for (const auto &col : local_columns) {
    std::vector<std::pair<int, double>> sorted_col(col.begin(), col.end());
    std::ranges::sort(sorted_col);
    nnzs.push_back(static_cast<int>(sorted_col.size()));
    for (const auto &[row, val] : sorted_col) {
      all_rows.push_back(row);
      all_vals.push_back(val);
    }
  }
  MPI_Send(nnzs.data(), static_cast<int>(nnzs.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);
  MPI_Send(all_rows.data(), static_cast<int>(all_rows.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);
  MPI_Send(all_vals.data(), static_cast<int>(all_vals.size()), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
}

void DorofeevICCSMatrixProductionMPI::BroadcastResults(OutType &output, int rank) {
  // Broadcast the result to all processes
  MPI_Bcast(&output.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&output.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int col_ptr_size = static_cast<int>(output.col_ptr.size());
  MPI_Bcast(&col_ptr_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    output.col_ptr.resize(col_ptr_size);
  }
  MPI_Bcast(output.col_ptr.data(), col_ptr_size, MPI_INT, 0, MPI_COMM_WORLD);
  int row_indices_size = static_cast<int>(output.row_indices.size());
  MPI_Bcast(&row_indices_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    output.row_indices.resize(row_indices_size);
  }
  MPI_Bcast(output.row_indices.data(), row_indices_size, MPI_INT, 0, MPI_COMM_WORLD);
  int values_size = static_cast<int>(output.values.size());
  MPI_Bcast(&values_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    output.values.resize(values_size);
  }
  MPI_Bcast(output.values.data(), values_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool DorofeevICCSMatrixProductionMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto [start_col, local_cols] = ComputeLocalColumnRange(rank, size, GetInput().second.cols);
  auto local_columns = ComputeLocalColumns(GetInput(), start_col, local_cols);

  GatherResultsOnRoot(GetOutput(), local_columns, size, rank);
  BroadcastResults(GetOutput(), rank);

  return true;
}

bool DorofeevICCSMatrixProductionMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dorofeev_i_ccs_matrix_production
