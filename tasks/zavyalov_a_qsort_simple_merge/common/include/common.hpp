#pragma once

#include <mpi.h>

#include <cstring>
#include <memory>
#include <tuple>

#include "task/include/task.hpp"

namespace zavyalov_a_qsort_simple_merge {
using InType = std::tuple<std::vector<double>>;  // сортируемый массив
using OutType = std::tuple<std::vector<double>>;  // результат сортировки
using TestType = std::tuple<size_t>;  // размер массива
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zavyalov_a_qsort_simple_merge
