#include "frolova_s_mult_int_trapez/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace frolova_s_mult_int_trapez {

FrolovaSMultIntTrapezMPI::FrolovaSMultIntTrapezMPI(const InType &in) : BaseTask(), result_(0.0) {
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

  // Защита от деления на 0 и пустых векторов
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
  if (rank == 0) {
    return !GetInput().limits.empty() && GetInput().number_of_intervals.size() == GetInput().limits.size();
  }
  return true;
}

bool FrolovaSMultIntTrapezMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
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

  int limits_size = static_cast<int>(limits_.size());
  MPI_Bcast(&limits_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    limits_.resize(limits_size);
    number_of_intervals_.resize(limits_size);
  }

  MPI_Bcast(limits_.data(), limits_size * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(number_of_intervals_.data(), limits_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  if (limits_size == 0) {
    return true;
  }

  unsigned long long total_points = 1;
  for (unsigned int val : number_of_intervals_) {
    total_points *= (val + 1);
  }

  unsigned int u_size = static_cast<unsigned int>(size);
  unsigned int base_delta = static_cast<unsigned int>(total_points / u_size);
  unsigned int remainder = static_cast<unsigned int>(total_points % u_size);

  unsigned int local_count = base_delta + (static_cast<unsigned int>(rank) < remainder ? 1 : 0);
  unsigned int local_start =
      static_cast<unsigned int>(rank) * base_delta + std::min(static_cast<unsigned int>(rank), remainder);

  double local_result = 0.0;
  for (unsigned int i = 0; i < local_count; i++) {
    std::vector<double> point = GetPointFromNumber(local_start + i);
    local_result += CalculationOfCoefficient(point) * GetInput().function(point);
  }

  MPI_Reduce(&local_result, &result_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (unsigned int i = 0; i < limits_.size(); i++) {
      result_ *= (limits_[i].second - limits_[i].first) / number_of_intervals_[i];
    }
    result_ /= std::pow(2, limits_.size());
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
