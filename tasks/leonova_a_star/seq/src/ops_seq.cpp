#include "leonova_a_star/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "leonova_a_star/common/include/common.hpp"

namespace leonova_a_star {

LeonovaAStarSEQ::LeonovaAStarSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;

  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  if (!matrix_a.empty() && !matrix_b.empty() && !matrix_b[0].empty()) {
    size_t rows = matrix_a.size();
    size_t cols = matrix_b[0].size();
    GetOutput().resize(rows, std::vector<int>(cols, 0));
  }
}

bool LeonovaAStarSEQ::PreProcessingImpl() {
  return true;
}

bool LeonovaAStarSEQ::ValidationImpl() {
  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  if (matrix_a.empty() || matrix_b.empty()) {
    return false;
  }

  if (matrix_a[0].empty() || matrix_b[0].empty()) {
    return false;
  }

  size_t rows_a = matrix_a.size();
  size_t cols_a = matrix_a[0].size();

  for (size_t index = 1; index < rows_a; index++) {
    if (matrix_a[index].size() != cols_a) {
      return false;
    }
  }

  size_t rows_b = matrix_b.size();
  size_t cols_b = matrix_b[0].size();

  for (size_t index = 1; index < rows_b; index++) {
    if (matrix_b[index].size() != cols_b) {
      return false;
    }
  }

  return cols_a == rows_b;
}

bool LeonovaAStarSEQ::RunImpl() {
  if (!ValidationImpl()) {
    return false;
  }

  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  size_t rows_a = matrix_a.size();
  size_t cols_a = matrix_a[0].size();
  size_t cols_b = matrix_b[0].size();

  if (GetOutput().size() != rows_a || (rows_a > 0 && GetOutput()[0].size() != cols_b)) {
    GetOutput().resize(rows_a, std::vector<int>(cols_b, 0));
  }

  for (size_t index = 0; index < rows_a; index++) {
    for (size_t jndex = 0; jndex < cols_b; jndex++) {
      int sum = 0;
      for (size_t kndex = 0; kndex < cols_a; kndex++) {
        sum += matrix_a[index][kndex] * matrix_b[kndex][jndex];
      }
      GetOutput()[index][jndex] = sum;
    }
  }

  return true;
}

bool LeonovaAStarSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace leonova_a_star
