#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>
#include <tuple>
#include <utility>
#include <exception>

#include "task/include/task.hpp"

namespace romanov_a_crs_product {

constexpr double kEps = 1e-6;

class Dense {
private:
  size_t n;
  std::vector<double> data;

public:
  Dense(size_t n) : n(n), data(n *n , 0.0) {}

  size_t size() const {
    return n;
  }

  double& operator()(size_t i, size_t j) {
    return data[(i * n) + j];
  }

  double operator()(size_t i, size_t j) const {
    return data[(i * n) + j];
  }

  bool operator==(const Dense& other) const {
    if (n != other.n) {
      return false;
    }
    for (size_t i = 0; i < n * n; ++i) {
        if (std::abs(data[i] - other.data[i]) > kEps) {
            return false;
        }
    }
    return true;
  }

  friend Dense operator*(const Dense& A, const Dense& B);
};

Dense operator*(const Dense& A, const Dense& B) {
  if (A.n != B.n) {
    throw std::runtime_error("Matrix dimensions do not match for multiplication!");
  }

  size_t n = A.n;
  Dense C(n);
  
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < n; ++k) {
        sum += A(i, k) * B(k, j);
      }
      C(i, j) = sum;
    }
  }
  return C;
}

class CRS {
private:
  size_t n;
  
  std::vector<double> value;
  std::vector<size_t> column;
  std::vector<size_t> row_index;

public:

  CRS(size_t n): n(n) {
    row_index.resize(n + 1, 0);
  }

  size_t size() const {
    return n;
  }

  size_t nnz() const {
    return value.size();
  }

  bool operator==(const CRS& other) const {
    if (n != other.n || column != other.column || row_index != other.row_index) {
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
    std::vector<size_t> new_row_index(n + 1, 0);

    std::vector<size_t> columns_count(n, 0);
    for (size_t i = 0; i < nnz; ++i) {
      ++columns_count[column[i]];
    }

    new_row_index[0] = 0;
    for (size_t i = 0; i < n; ++i) {
      new_row_index[i + 1] = new_row_index[i] + columns_count[i];
    }

    std::vector<size_t> offset = new_row_index;

    for (size_t row = 0; row < n; ++row) {
      size_t start = row_index[row];
      size_t end   = row_index[row + 1];

      for (size_t idx = start; idx < end; ++idx) {
        size_t col = column[idx];
        size_t pos = offset[col]++;

        new_value[pos]  = value[idx];
        new_column[pos] = row;
      }
    }

    value      = std::move(new_value);
    column     = std::move(new_column);
    row_index  = std::move(new_row_index);
  }

  CRS transpose() const {
    CRS result = *this;
    result.transpose();
    return result;
  }

  friend CRS operator*(const CRS& A, const CRS& B);
};

CRS operator*(const CRS& A, const CRS& B) {
  if (A.n != B.n) {
    throw std::runtime_error("Matrix dimensions do not match for multiplication!");
  }

  size_t n = A.n;

  CRS C(n);
  CRS BT = B.transpose();

  std::vector<std::pair<size_t, double>> row;

  for (size_t i = 0; i < n; ++i) {
    row.clear();

    size_t a_begin = A.row_index[i];
    size_t a_end   = A.row_index[i + 1];

    for (size_t j = 0; j < n; ++j) {
        size_t b_begin = BT.row_index[j];
        size_t b_end   = BT.row_index[j + 1];

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
          } 
          else if (a_col < b_col) {
            ++a_pos;
          } 
          else {
            ++b_pos;
          }
        }

        if (sum != 0.0) {
          row.emplace_back(j, sum);
        }
    }

    C.row_index[i] = C.column.size();
    for (auto& [col, val] : row) {
        C.column.push_back(col);
        C.value.push_back(val);
    }
  }

  C.row_index[n] = C.column.size();
  return C;
}

Dense ToDense(const CRS& crs) {
  size_t n = crs.size();
  Dense dense(n);

  for (size_t row = 0; row < n; ++row) {
    size_t start_p = crs.row_index[row];
    size_t end_p   = crs.row_index[row + 1];

    for (size_t idx = start_p; idx < end_p; ++idx) {
      dense(row, crs.column[idx]) = crs.value[idx];
    }
  }
  return dense;
}

CRS ToCRS(const Dense& A) {
  size_t n = A.size();
  CRS crs(n);

  crs.row_index[0] = static_cast<size_t>(0);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (std::abs(A(i, j)) > kEps) {
        crs.value.push_back(A(i,j));
        crs.column.push_back(j);
      }
    }
    crs.row_index[i + 1] = crs.value.size();
  }
  return crs;
}

using InType = std::tuple<size_t, CRS, CRS>;
using OutType = CRS;
using TestType = std::tuple<size_t, CRS, CRS, CRS>;
;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace romanov_a_crs_product
