#include "gauss_jordan/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "gauss_jordan/common/include/common.hpp"

namespace gauss_jordan {

namespace {
constexpr double EPSILON = 1e-12;

void ExchangeRows(std::vector<std::vector<double>> &augmented_matrix, int first_row, int second_row, int columns) {
  if (first_row == second_row) {
    return;
  }

  for (int col_idx = 0; col_idx < columns; ++col_idx) {
    std::swap(augmented_matrix[first_row][col_idx], augmented_matrix[second_row][col_idx]);
  }
}

inline bool IsNumericallyZero(double value) {
  return std::fabs(value) < EPSILON;
}

int LocatePivotIndex(const std::vector<std::vector<double>> &augmented_matrix, int start_row, int column,
                     int total_rows) {
  for (int row_idx = start_row; row_idx < total_rows; ++row_idx) {
    if (!IsNumericallyZero(augmented_matrix[row_idx][column])) {
      return row_idx;
    }
  }
  return -1;
}

void EliminateFromRow(std::vector<std::vector<double>> &augmented_matrix, int target_row, int source_row,
                      double elimination_coefficient, int columns) {
  if (IsNumericallyZero(elimination_coefficient)) {
    return;
  }

  for (int col_idx = 0; col_idx < columns; ++col_idx) {
    augmented_matrix[target_row][col_idx] -= elimination_coefficient * augmented_matrix[source_row][col_idx];
  }
}

void NormalizeRow(std::vector<std::vector<double>> &augmented_matrix, int row_index, double normalizer, int columns) {
  if (IsNumericallyZero(normalizer)) {
    return;
  }

  for (int col_idx = 0; col_idx < columns; ++col_idx) {
    augmented_matrix[row_index][col_idx] /= normalizer;
  }
}

void TransformToReducedRowEchelonForm(std::vector<std::vector<double>> &augmented_matrix, int equations_count,
                                      int augmented_columns) {
  int current_row = 0;

  for (int current_col = 0; current_col < augmented_columns - 1 && current_row < equations_count; current_col++) {
    // Выбор ведущего элемента (частичный выбор для улучшения устойчивости)
    int pivot_row = current_row;
    double max_val = std::abs(augmented_matrix[current_row][current_col]);

    for (int i = current_row + 1; i < equations_count; i++) {
      double val = std::abs(augmented_matrix[i][current_col]);
      if (val > max_val) {
        max_val = val;
        pivot_row = i;
      }
    }

    if (IsNumericallyZero(max_val)) {
      continue;
    }

    if (pivot_row != current_row) {
      ExchangeRows(augmented_matrix, current_row, pivot_row, augmented_columns);
    }

    double pivot_value = augmented_matrix[current_row][current_col];
    NormalizeRow(augmented_matrix, current_row, pivot_value, augmented_columns);

    for (int row_idx = 0; row_idx < equations_count; row_idx++) {
      if (row_idx == current_row) {
        continue;
      }

      double coefficient = augmented_matrix[row_idx][current_col];
      if (!IsNumericallyZero(coefficient)) {
        EliminateFromRow(augmented_matrix, row_idx, current_row, coefficient, augmented_columns);
      }
    }

    current_row++;
  }
}

bool IsZeroRow(const std::vector<double> &row, int columns_minus_one) {
  return std::all_of(row.begin(), row.begin() + columns_minus_one,
                     [](double element) { return IsNumericallyZero(element); });
}

bool HasInconsistentEquation(const std::vector<std::vector<double>> &augmented_matrix, int equations_count,
                             int augmented_columns) {
  for (int row_idx = 0; row_idx < equations_count; ++row_idx) {
    if (IsZeroRow(augmented_matrix[row_idx], augmented_columns - 1) &&
        !IsNumericallyZero(augmented_matrix[row_idx][augmented_columns - 1])) {
      return true;
    }
  }
  return false;
}

int ComputeMatrixRank(const std::vector<std::vector<double>> &augmented_matrix, int equations_count,
                      int augmented_columns) {
  int rank = 0;
  for (int row_idx = 0; row_idx < equations_count; ++row_idx) {
    if (!IsZeroRow(augmented_matrix[row_idx], augmented_columns - 1)) {
      ++rank;
    }
  }
  return rank;
}

std::vector<double> ComputeSolutionVector(const std::vector<std::vector<double>> &matrix, int n, int m) {
  std::vector<double> solution(m - 1, 0.0);
  int vars = m - 1;
  int rows_to_process = (n < vars) ? n : vars;

  for (int i = 0; i < rows_to_process; ++i) {
    int pivot_col = -1;

    for (int j = 0; j < vars; ++j) {
      if (std::abs(matrix[i][j]) > 1e-10) {
        pivot_col = j;
        break;
      }
    }
    if (pivot_col >= 0) {
      solution[pivot_col] = matrix[i][vars];
    } else {
      solution[i] = 0.0;
    }
  }

  return solution;
}

}  // namespace

GaussJordanSEQ::GaussJordanSEQ(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType matrix_copy(input);
  GetInput().swap(matrix_copy);
}

bool GaussJordanSEQ::ValidationImpl() {
  const auto &input_matrix = GetInput();

  if (input_matrix.empty()) {
    return true;
  }
  if (!GetOutput().empty()) {
    return false;
  }

  size_t column_count = input_matrix[0].size();
  if (column_count == 0) {
    return false;
  }

  size_t total_elements = 0;
  for (const auto &row : input_matrix) {
    total_elements += row.size();
    if (row.size() != column_count) {
      return false;
    }
  }

  return total_elements == (column_count * input_matrix.size());
}

bool GaussJordanSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool GaussJordanSEQ::RunImpl() {
  const auto &input_matrix = GetInput();

  if (input_matrix.empty()) {
    GetOutput() = std::vector<double>();
    return true;
  }

  std::vector<std::vector<double>> augmented_matrix = input_matrix;

  if (augmented_matrix.empty()) {
    GetOutput() = std::vector<double>();
    return true;
  }

  int equations_count = static_cast<int>(augmented_matrix.size());
  int augmented_columns = static_cast<int>(augmented_matrix[0].size());

  for (int i = 1; i < equations_count; ++i) {
    if (static_cast<int>(augmented_matrix[i].size()) != augmented_columns) {
      GetOutput() = std::vector<double>();
      return true;
    }
  }

  if (augmented_columns <= 1) {
    GetOutput() = std::vector<double>();
    return true;
  }

  bool is_zero_matrix = true;
  for (int i = 0; i < equations_count && is_zero_matrix; ++i) {
    for (int j = 0; j < augmented_columns && is_zero_matrix; ++j) {
      if (std::abs(augmented_matrix[i][j]) > 1e-12) {
        is_zero_matrix = false;
      }
    }
  }

  if (is_zero_matrix) {
    GetOutput() = std::vector<double>();
    return true;
  }

  TransformToReducedRowEchelonForm(augmented_matrix, equations_count, augmented_columns);

  if (HasInconsistentEquation(augmented_matrix, equations_count, augmented_columns)) {
    GetOutput() = std::vector<double>();
    return true;
  }

  int matrix_rank = ComputeMatrixRank(augmented_matrix, equations_count, augmented_columns);
  int variable_count = augmented_columns - 1;

  if (matrix_rank < variable_count && matrix_rank < equations_count) {
    GetOutput() = std::vector<double>();
    return true;
  }

  GetOutput() = ComputeSolutionVector(augmented_matrix, equations_count, augmented_columns);
  return true;
}

bool GaussJordanSEQ::PostProcessingImpl() {
  return true;
}
}  // namespace gauss_jordan
