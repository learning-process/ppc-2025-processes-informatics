#include "frolova_s_mult_int_trapez/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "frolova_s_mult_int_trapez/common/include/common.hpp"

namespace frolova_s_mult_int_trapez {

FrolovaSMultIntTrapezMPI::FrolovaSMultIntTrapezMPI(const InType &in)
    : BaseTask(), limits_(in.limits), number_of_intervals_(in.number_of_intervals), result_(0.0) {
  std::cout << "[MPI CONSTRUCTOR] Created FrolovaSMultIntTrapezMPI" << std::endl;
  std::cout << "[MPI CONSTRUCTOR] Input limits size: " << in.limits.size() << std::endl;
  std::cout << "[MPI CONSTRUCTOR] Input intervals size: " << in.number_of_intervals.size() << std::endl;
  std::cout << "[MPI CONSTRUCTOR] Function is " << (in.function ? "NOT null" : "NULL") << std::endl;
}

unsigned int FrolovaSMultIntTrapezMPI::CalculationOfCoefficient(const std::vector<double> &point) {
  auto degree = static_cast<unsigned int>(limits_.size());
  for (unsigned int i = 0; i < limits_.size(); i++) {
    if ((std::abs(limits_[i].first - point[i]) < 1e-9) || (std::abs(limits_[i].second - point[i]) < 1e-9)) {
      degree--;
    }
  }
  return static_cast<unsigned int>(std::pow(2.0, static_cast<double>(degree)));
}

void FrolovaSMultIntTrapezMPI::Recursive(std::vector<double> &point, unsigned int &definition, unsigned int divider,
                                         unsigned int variable) {
  if (variable > 0) {
    Recursive(point, definition, divider * (number_of_intervals_[variable] + 1), variable - 1);
  }
  if (variable < limits_.size() && number_of_intervals_[variable] != 0) {
    const auto quotient = static_cast<unsigned int>(definition) / divider;
    point[variable] = limits_[variable].first +
                      (static_cast<double>(quotient) * (limits_[variable].second - limits_[variable].first) /
                       static_cast<double>(number_of_intervals_[variable]));
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

  std::cout << "[MPI VALIDATION] Rank " << rank << ": Starting validation" << std::endl;

  if (rank == 0) {
    auto input = GetInput();

    std::cout << "[MPI VALIDATION] Rank 0 checking input:" << std::endl;
    std::cout << "[MPI VALIDATION]   limits.empty(): " << input.limits.empty() << std::endl;
    std::cout << "[MPI VALIDATION]   number_of_intervals.empty(): " << input.number_of_intervals.empty() << std::endl;

    if (input.limits.empty() || input.number_of_intervals.empty()) {
      std::cout << "[MPI VALIDATION] Rank 0: FAILED - empty limits or intervals" << std::endl;
      return false;
    }

    std::cout << "[MPI VALIDATION]   limits.size(): " << input.limits.size() << std::endl;
    std::cout << "[MPI VALIDATION]   intervals.size(): " << input.number_of_intervals.size() << std::endl;

    if (input.limits.size() != input.number_of_intervals.size()) {
      std::cout << "[MPI VALIDATION] Rank 0: FAILED - sizes don't match" << std::endl;
      return false;
    }

    std::cout << "[MPI VALIDATION]   function pointer: " << (input.function ? "NOT null" : "NULL") << std::endl;

    if (!input.function) {
      std::cout << "[MPI VALIDATION] Rank 0: FAILED - function is null" << std::endl;
      return false;
    }

    for (size_t i = 0; i < input.limits.size(); i++) {
      std::cout << "[MPI VALIDATION]   Limit " << i << ": [" << input.limits[i].first << ", " << input.limits[i].second
                << "], intervals: " << input.number_of_intervals[i] << std::endl;

      if (input.limits[i].first >= input.limits[i].second) {
        std::cout << "[MPI VALIDATION] Rank 0: FAILED - limit " << i << " has first >= second" << std::endl;
        return false;
      }

      if (input.number_of_intervals[i] == 0) {
        std::cout << "[MPI VALIDATION] Rank 0: FAILED - interval " << i << " is zero" << std::endl;
        return false;
      }
    }

    std::cout << "[MPI VALIDATION] Rank 0: PASSED validation" << std::endl;

    bool validation_result = true;
    MPI_Bcast(&validation_result, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    return validation_result;
  } else {
    bool validation_result = false;
    MPI_Bcast(&validation_result, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    std::cout << "[MPI VALIDATION] Rank " << rank
              << ": received validation result: " << (validation_result ? "PASSED" : "FAILED") << std::endl;
    return validation_result;
  }
}

bool FrolovaSMultIntTrapezMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::cout << "[MPI PRE_PROCESSING] Rank " << rank << ": Starting pre-processing" << std::endl;

  if (rank == 0) {
    auto input = GetInput();
    limits_ = input.limits;
    number_of_intervals_ = input.number_of_intervals;
    result_ = 0.0;

    std::cout << "[MPI PRE_PROCESSING] Rank 0: Initialized with limits size = " << limits_.size()
              << ", intervals size = " << number_of_intervals_.size() << std::endl;
  }

  std::cout << "[MPI PRE_PROCESSING] Rank " << rank << ": Finished pre-processing" << std::endl;
  return true;
}

bool FrolovaSMultIntTrapezMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int dims = (rank == 0) ? static_cast<int>(limits_.size()) : 0;
  MPI_Bcast(&dims, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    limits_.resize(dims);
    number_of_intervals_.resize(dims);
  }

  MPI_Bcast(limits_.data(), dims * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(number_of_intervals_.data(), dims, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  if (dims == 0) {
    return true;
  }

  std::uint64_t total_points = 1;
  for (unsigned int val : number_of_intervals_) {
    total_points *= (static_cast<std::uint64_t>(val) + 1);
  }

  auto u_size = static_cast<unsigned int>(size);
  auto base_delta = static_cast<unsigned int>(total_points / u_size);
  auto remainder = static_cast<unsigned int>(total_points % u_size);

  unsigned int local_count = base_delta + (std::cmp_less(static_cast<unsigned int>(rank), remainder) ? 1 : 0);
  unsigned int local_start =
      (static_cast<unsigned int>(rank) * base_delta) +
      (std::cmp_less(static_cast<unsigned int>(rank), remainder) ? static_cast<unsigned int>(rank) : remainder);

  double local_sum = 0.0;
  auto func = GetInput().function;

  if (func != nullptr) {
    for (unsigned int i = 0; i < local_count; i++) {
      std::vector<double> point = GetPointFromNumber(local_start + i);
      local_sum += static_cast<double>(CalculationOfCoefficient(point)) * func(point);
    }
  }

  MPI_Reduce(&local_sum, &result_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    double step_prod = 1.0;
    for (unsigned int i = 0; i < limits_.size(); i++) {
      step_prod *= (limits_[i].second - limits_[i].first) / static_cast<double>(number_of_intervals_[i]);
    }
    result_ *= step_prod / std::pow(2.0, static_cast<double>(limits_.size()));
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
