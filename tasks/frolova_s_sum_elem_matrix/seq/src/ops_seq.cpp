#include "frolova_s_sum_elem_matrix/seq/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

namespace frolova_s_sum_elem_matrix {

FrolovaSSumElemMatrixSEQ::FrolovaSSumElemMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool FrolovaSSumElemMatrixSEQ::ValidationImpl() {
  return true;
}

bool FrolovaSSumElemMatrixSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool FrolovaSSumElemMatrixSEQ::RunImpl() {
  const auto &matrix = GetInput();

  // Отладка для больших матриц
  std::cerr << "=== RUN IMPL DEBUG START ===\n";
  std::cerr << "matrix.size = " << matrix.size() << "\n";

  // Для больших матриц не печатаем все элементы
  if (matrix.size() > 10) {
    std::cerr << "Large matrix, checking first few rows...\n";
    if (!matrix.empty()) {
      std::cerr << "First row size: " << matrix[0].size() << "\n";
      if (!matrix[0].empty()) {
        std::cerr << "First element: " << matrix[0][0] << "\n";
      }
    }
  } else {
    // Для маленьких матриц печатаем все
    for (size_t row = 0; row < matrix.size(); ++row) {
      std::cerr << "ROW[" << row << "] size=" << matrix[row].size() << "\n";
    }
  }

  int64_t sum = 0;
  for (const auto &row : matrix) {
    sum += std::accumulate(row.begin(), row.end(), 0LL);
  }

  GetOutput() = sum;
  std::cerr << "computed sum = " << sum << "\n";
  std::cerr << "=== RUN IMPL DEBUG END ===\n";

  return true;
}

bool FrolovaSSumElemMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace frolova_s_sum_elem_matrix
