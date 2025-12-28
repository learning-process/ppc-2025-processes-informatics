#include "frolova_s_mult_int_trapez/mpi/include/ops_mpi.hpp"

#include <cmath>
#include <vector>

#include "mpi.h"

namespace frolova_s_mult_int_trapez {

FrolovaSMultIntTrapezMPI::FrolovaSMultIntTrapezMPI(const InType &in)
    : BaseTask(in), limits_(in.limits), number_of_intervals_(in.number_of_intervals), result_(0.0) {}

unsigned int FrolovaSMultIntTrapezMPI::CalculationOfCoefficient(const std::vector<double> &point) {
  unsigned int coefficient = 1;
  for (size_t i = 0; i < limits_.size(); ++i) {
    if ((std::abs(point[i] - limits_[i].first) < 1e-12) || (std::abs(point[i] - limits_[i].second) < 1e-12)) {
      coefficient *= 2;
    }
  }
  return coefficient;
}

[[gnu::noinline]]
void FrolovaSMultIntTrapezMPI::Recursive(std::vector<double> &point, unsigned int &definition, unsigned int divider,
                                         unsigned int variable) {
  if (variable > 0) {
    Recursive(point, definition, divider * (number_of_intervals_[variable] + 1), variable - 1);
  }

  point[variable] = limits_[variable].first +
                    (static_cast<double>(definition) / divider) *
                        ((limits_[variable].second - limits_[variable].first) / number_of_intervals_[variable]);
}

std::vector<double> FrolovaSMultIntTrapezMPI::GetPointFromNumber(unsigned int number) {
  std::vector<double> point(limits_.size());
  unsigned int definition = number;
  Recursive(point, definition, 1, static_cast<unsigned int>(limits_.size()) - 1);
  return point;
}

bool FrolovaSMultIntTrapezMPI::ValidationImpl() {
  if (limits_.size() != number_of_intervals_.size()) {
    return false;
  }

  for (size_t i = 0; i < limits_.size(); ++i) {
    if (limits_[i].first >= limits_[i].second) {
      return false;
    }
    if (number_of_intervals_[i] == 0) {
      return false;
    }
  }

  return true;
}

bool FrolovaSMultIntTrapezMPI::PreProcessingImpl() {
  result_ = 0.0;

  // Вычисление общего количества точек
  unsigned int total_points = 1;
  for (unsigned int number_of_interval : number_of_intervals_) {
    total_points *= (number_of_interval + 1);
  }

  // Рассчет размера блока для каждого процесса
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto u_size = static_cast<unsigned int>(size);
  unsigned int block_size = total_points / u_size;
  unsigned int remainder = total_points % u_size;

  // Начальный индекс для текущего процесса
  unsigned int start_index = rank * block_size + (rank < remainder ? rank : remainder);
  unsigned int end_index = start_index + block_size + (rank < remainder ? 1 : 0);

  // Локальные вычисления
  double local_result = 0.0;
  for (unsigned int i = start_index; i < end_index; ++i) {
    std::vector<double> point = GetPointFromNumber(i);
    unsigned int coefficient = CalculationOfCoefficient(point);
    double value = GetInput().function(point);
    local_result += static_cast<double>(coefficient) * value;
  }

  // Сбор результатов со всех процессов
  MPI_Reduce(&local_result, &result_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // Нормализация результата
  if (rank == 0 && !limits_.empty()) {
    result_ /= std::pow(2.0, static_cast<double>(limits_.size()));
  }

  return true;
}

bool FrolovaSMultIntTrapezMPI::RunImpl() {
  // Основная логика уже в PreProcessingImpl
  return true;
}

bool FrolovaSMultIntTrapezMPI::PostProcessingImpl() {
  // Пост-обработка если нужна
  return true;
}

}  // namespace frolova_s_mult_int_trapez
