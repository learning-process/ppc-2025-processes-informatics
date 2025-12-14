#include "gasenin_l_mult_int_mstep_trapez/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace gasenin_l_mult_int_mstep_trapez {

GaseninLMultIntMstepTrapezMPI::GaseninLMultIntMstepTrapezMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GaseninLMultIntMstepTrapezMPI::ValidationImpl() {
  return GetInput().n_steps > 0 && GetInput().x2 > GetInput().x1 && GetInput().y2 > GetInput().y1;
}

bool GaseninLMultIntMstepTrapezMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool GaseninLMultIntMstepTrapezMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  TaskData data = GetInput();
  MPI_Bcast(&data, sizeof(TaskData), MPI_BYTE, 0, MPI_COMM_WORLD);

  double hx = (data.x2 - data.x1) / data.n_steps;
  double hy = (data.y2 - data.y1) / data.n_steps;
  auto f = GetFunction(data.func_id);

  int delta = data.n_steps / size;
  int rem = data.n_steps % size;
  int start_i = rank * delta + std::min(rank, rem);
  int end_i = start_i + delta + (rank < rem ? 1 : 0);

  double local_sum = 0.0;
  for (int i = start_i; i < end_i; ++i) {
    double x = data.x1 + i * hx;
    double x_next = data.x1 + (i + 1) * hx;
    for (int j = 0; j < data.n_steps; ++j) {
      double y = data.y1 + j * hy;
      double y_next = data.y1 + (j + 1) * hy;

      local_sum += (f(x, y) + f(x_next, y) + f(x, y_next) + f(x_next, y_next));
    }
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_sum * hx * hy * 0.25;
  } else {
    GetOutput() = 0.0;
  }

  return true;
}

bool GaseninLMultIntMstepTrapezMPI::PostProcessingImpl() {
  return true;
}

TaskData GaseninLMultIntMstepTrapezMPI::ReadInputData(const std::string &filename, bool use_manual) {
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
