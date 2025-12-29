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
    if ((limits_[i].first == point[i]) || (limits_[i].second == point[i])) {
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
  point[variable] = limits_[variable].first + static_cast<double>(definition / divider) *
                                                  (limits_[variable].second - limits_[variable].first) /
                                                  number_of_intervals_[variable];
  definition = definition % divider;
}

std::vector<double> FrolovaSMultIntTrapezMPI::GetPointFromNumber(unsigned int number) {
  std::vector<double> point(limits_.size());
  unsigned int definition = number;
  Recursive(point, definition, 1, static_cast<unsigned int>(limits_.size()) - 1);
  return point;
}

bool FrolovaSMultIntTrapezMPI::ValidationImpl() {
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

  // Broadcast limits
  int limits_size = static_cast<int>(limits_.size());
  MPI_Bcast(&limits_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    limits_.resize(limits_size);
  }
  MPI_Bcast(limits_.data(), limits_size * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Broadcast number_of_intervals
  int intervals_size = static_cast<int>(number_of_intervals_.size());
  MPI_Bcast(&intervals_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    number_of_intervals_.resize(intervals_size);
  }
  MPI_Bcast(number_of_intervals_.data(), intervals_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  std::vector<unsigned int> count_of_points(size);
  std::vector<unsigned int> first_point_numbers(size);
  unsigned int local_count_of_points = 0;
  unsigned int local_first_point_numbers = 0;
  double local_result = 0.0;

  if (rank == 0) {
    unsigned int delta = 1;
    unsigned int current_number = 0;
    for (unsigned int i = 0; i < number_of_intervals_.size(); i++) {
      delta *= (number_of_intervals_[i] + 1);
    }

    unsigned int u_size = static_cast<unsigned int>(size);
    unsigned int remainder = delta % u_size;
    delta /= u_size;

    for (unsigned int i = 0; i < u_size - remainder; i++) {
      count_of_points[i] = delta;
      first_point_numbers[i] = current_number;
      current_number += delta;
    }

    delta++;
    for (unsigned int i = u_size - remainder; i < u_size; i++) {
      count_of_points[i] = delta;
      first_point_numbers[i] = current_number;
      current_number += delta;
    }
  }

  MPI_Scatter(count_of_points.data(), 1, MPI_UNSIGNED, &local_count_of_points, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Scatter(first_point_numbers.data(), 1, MPI_UNSIGNED, &local_first_point_numbers, 1, MPI_UNSIGNED, 0,
              MPI_COMM_WORLD);

  for (unsigned int i = 0; i < local_count_of_points; i++) {
    std::vector<double> point = GetPointFromNumber(local_first_point_numbers + i);
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
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

}  // namespace frolova_s_mult_int_trapez
