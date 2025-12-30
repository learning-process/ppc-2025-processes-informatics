#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace romanov_a_crs_product {

constexpr double kEps = 1e-6;

struct Dense {
  size_t n;
  size_t m;

  std::vector<double> data;

  Dense(size_t n, size_t m) : n(n), m(m), data(n * m, 0.0) {}
  Dense(size_t n) : n(n), m(n), data(n * n, 0.0) {}

  size_t getRows() const {
    return n;
  }

  size_t getCols() const {
    return m;
  }

  double &operator()(size_t i, size_t j) {
    return data[(i * m) + j];
  }

  double operator()(size_t i, size_t j) const {
    return data[(i * m) + j];
  }
};

struct CRS {
  size_t n;
  size_t m;

  std::vector<double> value;
  std::vector<size_t> column;
  std::vector<size_t> row_index;

  CRS() : n(0), m(0), row_index(1, 0) {}
  CRS(size_t n, size_t m) : n(n), m(m), row_index(n + 1, 0) {}
  CRS(size_t n) : n(n), m(n), row_index(n + 1, 0) {}

  size_t getRows() const {
    return n;
  }

  size_t getCols() const {
    return m;
  }

  size_t nnz() const {
    return value.size();
  }

  bool operator==(const CRS &other) const {
    if (n != other.n || m != other.m) {
      return false;
    }

    if (row_index != other.row_index || column != other.column) {
      return false;
    }

    for (size_t i = 0; i < value.size(); ++i) {
      if (std::abs(value[i] - other.value[i]) > kEps) {
        return false;
      }
    }

    return true;
  }

  void transpose() {
    size_t nnz_val = this->nnz();

    std::vector<double> new_value(nnz_val);
    std::vector<size_t> new_column(nnz_val);
    std::vector<size_t> new_row_index(m + 1, 0);

    for (size_t i = 0; i < nnz_val; ++i) {
      ++new_row_index[column[i] + 1];
    }

    for (size_t i = 1; i <= m; ++i) {
      new_row_index[i] += new_row_index[i - 1];
    }

    std::vector<size_t> offset = new_row_index;
    for (size_t row = 0; row < n; ++row) {
      for (size_t idx = row_index[row]; idx < row_index[row + 1]; ++idx) {
        size_t col = column[idx];
        size_t pos = offset[col]++;

        new_value[pos] = value[idx];
        new_column[pos] = row;
      }
    }

    value.swap(new_value);
    column.swap(new_column);
    row_index.swap(new_row_index);

    std::swap(n, m);
  }

  CRS getTransposed() const {
    CRS result = *this;
    result.transpose();
    return result;
  }

  void fillRandom(double density, unsigned seed = 0) {
    if (density < 0.0 || density > 1.0) {
      throw std::invalid_argument("Density must be within [0, 1]!");
    }

    value.clear();
    column.clear();
    row_index.assign(n + 1, 0);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_real_distribution<double> val(-1.0, 1.0);

    for (size_t row = 0; row < n; ++row) {
      row_index[row + 1] = row_index[row];
      for (size_t col = 0; col < m; ++col) {
        if (prob(gen) < density) {
          value.push_back(val(gen));
          column.push_back(col);
          ++row_index[row + 1];
        }
      }
    }
  }

  CRS ExtractRows(size_t start, size_t end) const {
    if (start >= end || start >= n) {
      return CRS(0, m);
    }

    end = std::min(end, n);
    size_t new_n = end - start;

    size_t nnz_start = row_index[start];
    size_t nnz_end = row_index[end];
    size_t nnz_count = nnz_end - nnz_start;

    CRS result(new_n, m);
    result.value.resize(nnz_count);
    result.column.resize(nnz_count);
    result.row_index.resize(new_n + 1);

    std::copy(value.begin() + nnz_start, value.begin() + nnz_end, result.value.begin());
    std::copy(column.begin() + nnz_start, column.begin() + nnz_end, result.column.begin());

    for (size_t i = 0; i <= new_n; ++i) {
      result.row_index[i] = row_index[start + i] - nnz_start;
    }

    return result;
  }

  static CRS ConcatRows(const std::vector<CRS> &parts) {
    if (parts.empty()) {
      return CRS();
    }

    size_t total_n = 0;
    size_t total_nnz = 0;
    size_t m = parts[0].m;

    for (const auto &part : parts) {
      total_n += part.n;
      total_nnz += part.nnz();
    }

    CRS result(total_n, m);
    result.value.resize(total_nnz);
    result.column.resize(total_nnz);
    result.row_index.resize(total_n + 1);

    size_t row_offset = 0;
    size_t nnz_offset = 0;

    for (const auto &part : parts) {
      std::copy(part.value.begin(), part.value.end(), result.value.begin() + nnz_offset);
      std::copy(part.column.begin(), part.column.end(), result.column.begin() + nnz_offset);

      for (size_t i = 0; i <= part.n; ++i) {
        result.row_index[row_offset + i] = part.row_index[i] + nnz_offset;
      }

      row_offset += part.n;
      nnz_offset += part.nnz();
    }

    return result;
  }
};

inline CRS operator*(const CRS &A, const CRS &B) {
  if (A.getCols() != B.getRows()) {
    throw std::runtime_error("Matrix dimensions do not match for multiplication!");
  }

  size_t n_rows = A.getRows();
  size_t n_cols = B.getCols();

  CRS C(n_rows, n_cols);
  CRS BT = B.getTransposed();

  std::vector<std::pair<size_t, double>> row;

  for (size_t i = 0; i < n_rows; ++i) {
    row.clear();
    size_t a_begin = A.row_index[i];
    size_t a_end = A.row_index[i + 1];

    for (size_t j = 0; j < n_cols; ++j) {
      size_t b_begin = BT.row_index[j];
      size_t b_end = BT.row_index[j + 1];

      size_t a_pos = a_begin;
      size_t b_pos = b_begin;

      double sum = 0.0;
      while (a_pos < a_end && b_pos < b_end) {
        size_t a_col = A.column[a_pos];
        size_t b_col = BT.column[b_pos];

        if (a_col == b_col) {
          sum += A.value[a_pos] * BT.value[b_pos];
          ++a_pos;
          ++b_pos;
        } else if (a_col < b_col) {
          ++a_pos;
        } else {
          ++b_pos;
        }
      }

      if (std::abs(sum) > kEps) {
        row.emplace_back(j, sum);
      }
    }

    C.row_index[i] = C.column.size();

    for (auto &[col, val] : row) {
      C.column.push_back(col);
      C.value.push_back(val);
    }
  }

  C.row_index[n_rows] = C.column.size();

  return C;
}

inline CRS ToCRS(const Dense &A) {
  size_t rows = A.getRows();
  size_t cols = A.getCols();
  CRS crs(rows, cols);

  for (size_t i = 0; i < rows; ++i) {
    crs.row_index[i + 1] = crs.row_index[i];
    for (size_t j = 0; j < cols; ++j) {
      if (std::abs(A(i, j)) > kEps) {
        crs.value.push_back(A(i, j));
        crs.column.push_back(j);
        ++crs.row_index[i + 1];
      }
    }
  }
  return crs;
}

using InType = std::tuple<CRS, CRS>;
using OutType = CRS;
using TestType = std::tuple<CRS, CRS, CRS>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace romanov_a_crs_product
