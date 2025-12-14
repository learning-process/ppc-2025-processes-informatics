#include "gasenin_l_mult_int_mstep_trapez/seq/include/ops_seq.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace gasenin_l_mult_int_mstep_trapez {

GaseninLMultIntMstepTrapezSEQ::GaseninLMultIntMstepTrapezSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GaseninLMultIntMstepTrapezSEQ::ValidationImpl() {
  return GetInput().n_steps > 0 && GetInput().x2 > GetInput().x1 && GetInput().y2 > GetInput().y1;
}

bool GaseninLMultIntMstepTrapezSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool GaseninLMultIntMstepTrapezSEQ::RunImpl() {
  auto data = GetInput();
  auto f = GetFunction(data.func_id);
  double hx = (data.x2 - data.x1) / data.n_steps;
  double hy = (data.y2 - data.y1) / data.n_steps;

  std::vector<double> x_vals(data.n_steps + 1);
  std::vector<double> y_vals(data.n_steps + 1);

#pragma omp parallel for if (data.n_steps > 1000)
  for (int i = 0; i <= data.n_steps; i++) {
    x_vals[i] = data.x1 + i * hx;
    y_vals[i] = data.y1 + i * hy;
  }

  double res = 0.0;
  double hxhy = hx * hy;

#pragma omp parallel for reduction(+ : res) if (data.n_steps > 500)
  for (int i = 0; i < data.n_steps; i++) {
    for (int j = 0; j < data.n_steps; j++) {
      double x_i = x_vals[i];
      double x_i1 = x_vals[i + 1];
      double y_j = y_vals[j];
      double y_j1 = y_vals[j + 1];

      double f00 = f(x_i, y_j);
      double f10 = f(x_i1, y_j);
      double f01 = f(x_i, y_j1);
      double f11 = f(x_i1, y_j1);

      res += (f00 + f10 + f01 + f11) * 0.25;
    }
  }

  GetOutput() = res * hxhy;

  double correction = 0.5 * (f(data.x1, data.y1) + f(data.x2, data.y1) + f(data.x1, data.y2) + f(data.x2, data.y2));
  GetOutput() -= correction * hxhy * 0.25;

  return true;
}

bool GaseninLMultIntMstepTrapezSEQ::PostProcessingImpl() {
  return true;
}

TaskData GaseninLMultIntMstepTrapezSEQ::ReadInputData(const std::string &filename, bool use_manual) {
  TaskData data;

  if (!use_manual && !filename.empty()) {
    std::ifstream file(filename);
    if (file.is_open()) {
      file >> data.n_steps >> data.func_id >> data.x1 >> data.x2 >> data.y1 >> data.y2;
      file.close();

      if (data.n_steps <= 0 || data.x2 <= data.x1 || data.y2 <= data.y1) {
        throw std::invalid_argument("Некорректные данные в файле");
      }
      return data;
    } else {
      std::cout << "Файл не найден, переключаюсь на ручной ввод" << std::endl;
    }
  }

  std::cout << "Введите параметры для вычисления интеграла:" << std::endl;
  std::cout << "Количество шагов: ";
  std::cin >> data.n_steps;

  std::cout << "Выберите функцию (0: x+y, 1: x^2+y^2, 2: sin(x)*cos(y)): ";
  std::cin >> data.func_id;

  std::cout << "Границы по X (x1 x2): ";
  std::cin >> data.x1 >> data.x2;

  std::cout << "Границы по Y (y1 y2): ";
  std::cin >> data.y1 >> data.y2;

  if (data.n_steps <= 0 || data.x2 <= data.x1 || data.y2 <= data.y1) {
    throw std::invalid_argument("Некорректные входные данные");
  }

  return data;
}

}  // namespace gasenin_l_mult_int_mstep_trapez
