#include "frolova_s_mult_int_trapez/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace frolova_s_mult_int_trapez {

FrolovaSMultIntTrapezMPI::FrolovaSMultIntTrapezMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

unsigned int FrolovaSMultIntTrapezMPI::CalculationOfCoefficient(const std::vector<double> &point) {
  unsigned int degree = limits.size();
  for (unsigned int i = 0; i < limits.size(); i++) {
    if ((limits[i].first == point[i]) || (limits[i].second == point[i])) {
      degree--;
    }
  }

  return pow(2, degree);
}

void FrolovaSMultIntTrapezMPI::Recursive(std::vector<double> &point, unsigned int &definition, unsigned int divider,
                                         unsigned int variable) {
  if (variable > 0) {
    Recursive(point, definition, divider * (number_of_intervals[variable] + 1), variable - 1);
  }
  _point[variable] = limits[variable].first + definition / divider *
                                                  (limits[variable].second - limits[variable].first) /
                                                  number_of_intervals[variable];
  definition = definition % divider;
}

std::vector<double> FrolovaSMultIntTrapezMPI::GetPointFromNumber(unsigned int number) {
  std::vector<double> point(limits.size());
  unsigned int definition = number;
  Recursive(point, definition, 1, limits.size() - 1);

  return point;
}

bool FrolovaSMultIntTrapezMPI::ValidationImpl() {
  return true;
}

bool FrolovaSMultIntTrapezMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    limits = GetInput().limits;
    number_of_intervals = GetInput().number_of_intervals;
    result = 0.0;
  }

  return true;
}

bool FrolovaSMultIntTrapezMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Broadcast limits
  int limits_size = limits.size();
  MPI_Bcast(&limits_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    limits.resize(limits_size);
  }
  MPI_Bcast(limits.data(), limits_size * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Broadcast number_of_intervals
  int intervals_size = number_of_intervals.size();
  MPI_Bcast(&intervals_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    number_of_intervals.resize(intervals_size);
  }
  MPI_Bcast(number_of_intervals.data(), intervals_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  std::vector<unsigned int> count_of_points(size);
  std::vector<unsigned int> first_point_numbers(size);
  unsigned int local_count_of_points = 0;
  unsigned int local_first_point_numbers = 0;
  double local_result = 0.0;

  if (rank == 0) {
    unsigned int delta = 1;
    unsigned int current_number = 0;
    for (unsigned int i = 0; i < number_of_intervals.size(); i++) {
      delta *= (number_of_intervals[i] + 1);
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

  MPI_Reduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (unsigned int i = 0; i < limits.size(); i++) {
      result *= (limits[i].second - limits[i].first) / number_of_intervals[i];
    }
    result /= pow(2, limits.size());
  }

  return true;
}

bool FrolovaSMultIntTrapezMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() = result;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

}  // namespace frolova_s_mult_int_trapez
