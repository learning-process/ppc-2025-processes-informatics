#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace shekhirev_v_cg_method {

struct InputData {
  std::vector<double> A;  // Матрица N*N, развернутая в вектор
  std::vector<double> b;  // Вектор правой части
  int n;                  // Размерность системы

  InputData() : n(0) {}
  InputData(const std::vector<double> &matrix, const std::vector<double> &rhs, int size) : A(matrix), b(rhs), n(size) {}
};

// Выходные данные - вектор решения X
using InType = InputData;
using OutType = std::vector<double>;

// Параметры теста: InputData, Ожидаемый результат (для проверки), Имя теста
using TestType = std::tuple<InputData, std::vector<double>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shekhirev_v_cg_method
