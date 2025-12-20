#include "chyokotov_a_seidel_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "chyokotov_a_seidel_method/common/include/common.hpp"

namespace chyokotov_a_seidel_method {

ChyokotovASeidelMethodSEQ::ChyokotovASeidelMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().first.clear();
  GetInput().first.reserve(in.first.size());
  for (const auto &row : in.first) {
    GetInput().first.push_back(row);
  }
  GetInput().second = in.second;
  GetOutput().clear();
}

bool ChyokotovASeidelMethodSEQ::ValidationImpl() {
  const auto &matrix = GetInput().first;
  if (matrix.empty()) {
    return true;
  }
  if (GetInput().second.size() != matrix.size()) {
    return false;
  }

  size_t n = matrix[0].size();
  for (const auto &row : matrix) {
    if (row.size() != n) {
      return false;
    }
  }

  for (size_t i = 0; i < n; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < n; j++) {
      if (i != j) {
        sum += std::abs(matrix[i][j]);
      }
    }
    if (std::abs(matrix[i][i]) <= sum) {
      return false;
    }
  }

  return true;
}

bool ChyokotovASeidelMethodSEQ::PreProcessingImpl() {
  if (GetInput().first.empty()) {
    return true;
  }
  GetOutput().resize(GetInput().second.size(), 0.0);
  return true;
}

bool ChyokotovASeidelMethodSEQ::RunImpl() {
  const auto &matrix = GetInput().first;
  if (matrix.empty()) {
    return true;
  }
  const auto &vec = GetInput().second;
  auto &output = GetOutput();
  int n = matrix.size();

  const int maxiter = 1000;

  for (int it = 0; it < maxiter; it++) {
    for (int i = 0; i < n; i++) {
      double new_x = vec[i];

      for (int j = 0; j < n; j++) {
        if (i != j) {
          new_x -= matrix[i][j] * output[j];
        }
      }

      output[i] = new_x / matrix[i][i];
    }

    double max_error = 0.0;
    for (int i = 0; i < n; i++) {
      double error = vec[i];
      for (int j = 0; j < n; j++) {
        error -= matrix[i][j] * output[j];
      }
      if (std::abs(error) > max_error) {
        max_error = std::abs(error);
      }
    }

    if (max_error < 0.000001) {
      break;
    }
  }

  return true;
}

bool ChyokotovASeidelMethodSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace chyokotov_a_seidel_method
