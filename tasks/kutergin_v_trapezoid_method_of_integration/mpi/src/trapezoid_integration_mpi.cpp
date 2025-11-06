#include "../include/trapezoid_integration_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <iostream>

#include "util/include/util.hpp"

namespace kutergin_v_trapezoid_mpi {

double func(double x)  // интегрируемая функция для примера
{
  return x * x;
}

TrapezoidIntegrationMPI::TrapezoidIntegrationMPI(const kutergin_v_trapezoid_seq::InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());  // установка типа задачи
  GetInput() = in;                       // сохранение входных данных
  GetOutput() = 0.0;                     // инициализация выходных данных
}

bool TrapezoidIntegrationMPI::ValidationImpl() {
  /*
  if (GetInput().b <= GetInput().a ||
      GetInput().n <= 0)  // проверка b > a (границ интегрирования) и n > 0 (число разбиений)
  {
    return false;
  }
  if (ppc::util::IsUnderMpirun()) {
    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);  // получение общего числа процессов
    if (GetInput().n % process_count != 0)          // число разбиений делится нацело на число процессов
    {
      return false;
    }
  }
  return true;
  */
  return (GetInput().a < GetInput().b) && (GetInput().n > 0);
}

bool TrapezoidIntegrationMPI::PreProcessingImpl() {
  int process_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);  // получение ранга процесса

  kutergin_v_trapezoid_seq::InType tmp_input = GetInput();  // создание хранилища для данных со всех процессов

  MPI_Bcast(&tmp_input, sizeof(tmp_input), MPI_BYTE, 0,
            MPI_COMM_WORLD);  // получение остальными процессами исходных данных от процесса с рангом 0

  GetInput() = tmp_input;

  return true;
}

bool TrapezoidIntegrationMPI::RunImpl() {
  int process_rank;
  int process_count;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

  double a = GetInput().a;
  double b = GetInput().b;
  int n = GetInput().n;
  double h = (b - a) / n;

  const int base_n = n / process_count;     // целое часть от деления числа разбиений на число процессов
  const int remainder = n % process_count;  // остаток от деления числа разбиений на число процессов

  const int local_n = base_n + (process_rank < remainder ? 1 : 0);  // количество разбиений (трапеций) на один процесс

  int start_index;
  if (process_rank < remainder) {
    start_index = process_rank * (base_n + 1);
  } else {
    start_index = remainder * (base_n + 1) + (process_rank - remainder) * base_n;
  }

  double local_a = a + start_index * h;  // начало отрезка для текущего процесса

  // локальные вычисления
  double local_sum = 0.0;
  if (local_n > 0) {
    local_sum = (func(local_a) + func(local_a + local_n * h)) / 2.0;
  }

  for (int i = 1; i < local_n; ++i) {
    local_sum += func(local_a + i * h);
  }

  // агрегация
  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (process_rank == 0) {
    GetOutput() = global_sum * h;
  }

  return true;
}

bool TrapezoidIntegrationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_v_trapezoid_mpi
