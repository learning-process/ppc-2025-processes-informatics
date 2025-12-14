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

  TaskData data;

  // Только процесс 0 имеет исходные данные
  if (rank == 0) {
    data = GetInput();
  }

  // Рассылаем данные всем процессам
  MPI_Bcast(&data.n_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data.func_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data.x1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data.x2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data.y1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data.y2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double hx = (data.x2 - data.x1) / data.n_steps;
  double hy = (data.y2 - data.y1) / data.n_steps;
  auto f = GetFunction(data.func_id);

  // Распределение работы: каждый процесс обрабатывает свою часть строк
  int total_steps = data.n_steps;
  int steps_per_process = total_steps / size;
  int remainder = total_steps % size;

  int start_i = rank * steps_per_process + std::min(rank, remainder);
  int end_i = start_i + steps_per_process + (rank < remainder ? 1 : 0);

  // Локальная сумма на каждом процессе
  double local_sum = 0.0;

  for (int i = start_i; i < end_i; ++i) {
    double x_i = data.x1 + i * hx;
    double x_i1 = data.x1 + (i + 1) * hx;

    for (int j = 0; j < total_steps; ++j) {
      double y_j = data.y1 + j * hy;
      double y_j1 = data.y1 + (j + 1) * hy;

      double f00 = f(x_i, y_j);
      double f10 = f(x_i1, y_j);
      double f01 = f(x_i, y_j1);
      double f11 = f(x_i1, y_j1);

      local_sum += (f00 + f10 + f01 + f11);
    }
  }

  // Собираем все частичные суммы на процессе 0
  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // Процесс 0 вычисляет финальный результат
  double result = 0.0;
  if (rank == 0) {
    result = global_sum * (hx * hy) / 4.0;
  }

  // Рассылаем результат всем процессам (важно для тестов!)
  MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Все процессы сохраняют результат
  GetOutput() = result;

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
