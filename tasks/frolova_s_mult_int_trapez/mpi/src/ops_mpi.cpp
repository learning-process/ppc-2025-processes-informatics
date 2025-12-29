#include "frolova_s_mult_int_trapez/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace frolova_s_mult_int_trapez {

FrolovaSMultIntTrapezMPI::FrolovaSMultIntTrapezMPI(const InType &in) : BaseTask(), result_(0.0) {
  // В тестах данные приходят только в объект на rank 0
  limits_ = in.limits;
  number_of_intervals_ = in.number_of_intervals;
}

unsigned int FrolovaSMultIntTrapezMPI::CalculationOfCoefficient(const std::vector<double> &point) {
  unsigned int degree = static_cast<unsigned int>(limits_.size());
  for (unsigned int i = 0; i < limits_.size(); i++) {
    if ((std::abs(limits_[i].first - point[i]) < 1e-9) || (std::abs(limits_[i].second - point[i]) < 1e-9)) {
      degree--;
    }
  }
  return static_cast<unsigned int>(std::pow(2, degree));
}

void FrolovaSMultIntTrapezMPI::Recursive(std::vector<double> &point, unsigned int &definition, unsigned int divider,
                                         unsigned int variable) {
  if (variable > 0) {
    Recursive(point, definition, divider * (number_of_intervals_[variable] + 1), variable - 1);
  }
  if (variable < limits_.size() && number_of_intervals_[variable] != 0) {
    point[variable] = limits_[variable].first + static_cast<double>(definition / divider) *
                                                    (limits_[variable].second - limits_[variable].first) /
                                                    number_of_intervals_[variable];
  }
  definition = definition % divider;
}

std::vector<double> FrolovaSMultIntTrapezMPI::GetPointFromNumber(unsigned int number) {
  std::vector<double> point(limits_.size());
  unsigned int definition = number;
  if (!limits_.empty()) {
    Recursive(point, definition, 1, static_cast<unsigned int>(limits_.size()) - 1);
  }
  return point;
}

bool FrolovaSMultIntTrapezMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Только ранг 0 может реально проверить GetInput()
  if (rank == 0) {
    return !GetInput().limits.empty() && GetInput().limits.size() == GetInput().number_of_intervals.size();
  }
  // Остальные процессы соглашаются, так как они получат данные в RunImpl
  return true;
}

bool FrolovaSMultIntTrapezMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    // Гарантируем, что локальные копии на rank 0 актуальны перед Bcast
    limits_ = GetInput().limits;
    number_of_intervals_ = GetInput().number_of_intervals;
    result_ = 0.0;
  }
  return true;
}

bool FrolovaSMultIntTrapezMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // 1. Раздаем размеры данных от ранга 0 всем остальным
  int dims = (rank == 0) ? static_cast<int>(limits_.size()) : 0;
  MPI_Bcast(&dims, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    limits_.resize(dims);
    number_of_intervals_.resize(dims);
  }

  // 2. Раздаем сами значения границ и интервалов
  MPI_Bcast(limits_.data(), dims * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(number_of_intervals_.data(), dims, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  if (dims == 0) {
    return true;
  }

  // 3. Считаем сетку точек
  unsigned long long total_points = 1;
  for (unsigned int val : number_of_intervals_) {
    total_points *= (static_cast<unsigned long long>(val) + 1);
  }

  // 4. Распределяем индексы точек между потоками
  unsigned int u_size = static_cast<unsigned int>(size);
  unsigned int base_delta = static_cast<unsigned int>(total_points / u_size);
  unsigned int remainder = static_cast<unsigned int>(total_points % u_size);

  unsigned int local_count = base_delta + (static_cast<unsigned int>(rank) < remainder ? 1 : 0);
  unsigned int local_start =
      static_cast<unsigned int>(rank) * base_delta + (static_cast<unsigned int>(rank) < remainder ? rank : remainder);

  // 5. Каждый процесс считает свою часть суммы
  double local_sum = 0.0;
  auto func = GetInput().function;

  if (func != nullptr) {
    for (unsigned int i = 0; i < local_count; i++) {
      std::vector<double> point = GetPointFromNumber(local_start + i);
      local_sum += static_cast<double>(CalculationOfCoefficient(point)) * func(point);
    }
  }

  // 6. Собираем все локальные суммы обратно в result_ на ранг 0
  MPI_Reduce(&local_sum, &result_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // 7. Финальный расчет на ранге 0
  if (rank == 0) {
    double step_prod = 1.0;
    for (unsigned int i = 0; i < limits_.size(); i++) {
      step_prod *= (limits_[i].second - limits_[i].first) / number_of_intervals_[i];
    }
    result_ *= step_prod / std::pow(2, limits_.size());
  }

  return true;
}

bool FrolovaSMultIntTrapezMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput() = result_;
  }
  return true;
}

}  // namespace frolova_s_mult_int_trapez
