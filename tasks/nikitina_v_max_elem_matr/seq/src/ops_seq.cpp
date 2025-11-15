#include "nikitina_v_max_elem_matr/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <stdexcept>

namespace nikitina_v_max_elem_matr {

MaxElementMatrSEQ::MaxElementMatrSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MaxElementMatrSEQ::ValidationImpl() {
  const auto &in_ = GetInput();
  if (in_.size() < 2) {
    return false;
  }
  rows = in_[0];
  cols = in_[1];
  // ================== ИСПРАВЛЕНИЕ ЗДЕСЬ ==================
  // Разрешаем нулевые размеры (это валидная пустая матрица),
  // но запрещаем отрицательные.
  if (rows < 0 || cols < 0 || static_cast<size_t>(rows * cols) != in_.size() - 2) {
    return false;
  }
  // =======================================================
  return true;
}

bool MaxElementMatrSEQ::PreProcessingImpl() {
  matrix_.clear();
  // Если матрица не пустая, копируем данные
  if (rows > 0 && cols > 0) {
    matrix_.reserve(rows * cols);
    const auto &in_ = GetInput();
    std::copy(in_.begin() + 2, in_.end(), std::back_inserter(matrix_));
  }
  max_val = INT_MIN;
  return true;
}

bool MaxElementMatrSEQ::RunImpl() {
  if (matrix_.empty()) {
    max_val = INT_MIN;
    return true;
  }
  max_val = matrix_[0];
  for (size_t i = 1; i < matrix_.size(); ++i) {
    if (matrix_[i] > max_val) {
      max_val = matrix_[i];
    }
  }
  return true;
}

bool MaxElementMatrSEQ::PostProcessingImpl() {
  GetOutput() = max_val;
  return true;
}

}  // namespace nikitina_v_max_elem_matr
