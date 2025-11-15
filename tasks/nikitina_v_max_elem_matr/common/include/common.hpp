#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"

namespace nikitina_v_max_elem_matr {

// Вход: std::vector<int>, где первые 2 элемента - размеры, остальные - матрица
using InType = std::vector<int>;
// Выход: int - максимальный элемент
using OutType = int;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nikitina_v_max_elem_matr
