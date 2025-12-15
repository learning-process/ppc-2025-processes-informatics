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

  double sum = 0.0;

  for (int i = 1; i < data.n_steps; i++) {
    double x = data.x1 + i * hx;
    for (int j = 1; j < data.n_steps; j++) {
      double y = data.y1 + j * hy;
      sum += f(x, y);
    }
  }

  for (int i = 1; i < data.n_steps; i++) {
    double x = data.x1 + i * hx;
    sum += 0.5 * f(x, data.y1);
    sum += 0.5 * f(x, data.y2);
  }

  for (int j = 1; j < data.n_steps; j++) {
    double y = data.y1 + j * hy;
    sum += 0.5 * f(data.x1, y);
    sum += 0.5 * f(data.x2, y);
  }

  sum += 0.25 * f(data.x1, data.y1);
  sum += 0.25 * f(data.x2, data.y1);
  sum += 0.25 * f(data.x1, data.y2);
  sum += 0.25 * f(data.x2, data.y2);

  GetOutput() = sum * hx * hy;

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
