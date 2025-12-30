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

  CRS(size_t n, size_t m) : n(n), m(m) {
    row_index.resize(n + 1, 0);
  }

  CRS(size_t n) : n(n), m(n) {
    row_index.resize(n + 1, 0);
  }

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
    if (n != other.n || m != other.m || column != other.column || row_index != other.row_index) {
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
    size_t nnz = this->nnz();

    std::vector<double> new_value(nnz);
    std::vector<size_t> new_column(nnz);
    std::vector<size_t> new_row_index(m + 1, 0);

    std::vector<size_t> columns_count(m, static_cast<size_t>(0));
    for (size_t i = 0; i < nnz; ++i) {
      ++columns_count[column[i]];
    }

    new_row_index[0] = 0;
    for (size_t i = 0; i < m; ++i) {
      new_row_index[i + 1] = new_row_index[i] + columns_count[i];
    }

    std::vector<size_t> offset = new_row_index;

    for (size_t row = 0; row < n; ++row) {
      size_t start = row_index[row];
      size_t end = row_index[row + static_cast<size_t>(1)];

      for (size_t idx = start; idx < end; ++idx) {
        size_t col = column[idx];
        size_t pos = offset[col]++;

        new_value[pos] = value[idx];
        new_column[pos] = row;
      }
    }

    value = std::move(new_value);
    column = std::move(new_column);
    row_index = std::move(new_row_index);

    std::swap(n, m);
  }

  CRS transpose() const {
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

    value.reserve(static_cast<size_t>(n * m * density));
    column.reserve(static_cast<size_t>(n * m * density));

    for (size_t row = 0; row < n; ++row) {
      std::vector<size_t> cols;
      for (size_t col = 0; col < m; ++col) {
        if (prob(gen) < density) {
          cols.push_back(col);
        }
      }
      for (size_t col : cols) {
        column.push_back(col);
        value.push_back(val(gen));
      }

      row_index[row + 1] = column.size();
    }
  }
};

inline CRS operator*(const CRS &A, const CRS &B) {
  if (A.getCols() != B.getRows()) {
    throw std::runtime_error("Matrix dimensions do not match for multiplication!");
  }

  size_t n_rows = A.getRows();
  size_t n_cols = B.getCols();

  CRS C(n_rows, n_cols);
  CRS BT = B.transpose();

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

      if (sum != 0.0) {
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
    for (size_t j = 0; j < cols; ++j) {
      if (std::abs(A(i, j)) > kEps) {
        crs.value.push_back(A(i, j));
        crs.column.push_back(j);
      }
    }
    crs.row_index[i + 1] = crs.value.size();
  }
  return crs;
}

using InType = std::tuple<CRS, CRS>;
using OutType = CRS;
using TestType = std::tuple<CRS, CRS, CRS>;
;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace romanov_a_crs_product
