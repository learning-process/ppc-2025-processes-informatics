#include "frolova_s_mult_int_trapez/seq/include/ops_seq.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include "frolova_s_mult_int_trapez/common/include/common.hpp"

namespace frolova_s_mult_int_trapez {

FrolovaSMultIntTrapezSEQ::FrolovaSMultIntTrapezSEQ(const InType &in)
    : limits_(in.limits), number_of_intervals_(in.number_of_intervals), result_(0.0) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;

  // std::cout << "[SEQ CONSTRUCTOR] Created FrolovaSMultIntTrapezSEQ" << std::endl;
  // std::cout << "[SEQ CONSTRUCTOR] Input limits size: " << in.limits.size() << std::endl;
  // std::cout << "[SEQ CONSTRUCTOR] Input intervals size: " << in.number_of_intervals.size() << std::endl;
  // std::cout << "[SEQ CONSTRUCTOR] Function is " << (in.function ? "NOT null" : "NULL") << std::endl;
}

unsigned int FrolovaSMultIntTrapezSEQ::CalculationOfCoefficient(const std::vector<double> &point) {
  auto degree = static_cast<unsigned int>(limits_.size());
  for (unsigned int i = 0; i < limits_.size(); i++) {
    if ((limits_[i].first == point[i]) || (limits_[i].second == point[i])) {
      degree--;
    }
  }
  return static_cast<unsigned int>(std::pow(2.0, static_cast<double>(degree)));
}

void FrolovaSMultIntTrapezSEQ::Recursive(std::vector<double> &point, unsigned int &definition, unsigned int divider,
                                         unsigned int variable) {
  if (variable > 0) {
    Recursive(point, definition, divider * (number_of_intervals_[variable] + 1), variable - 1);
  }
  const auto quotient = static_cast<unsigned int>(definition) / divider;
  point[variable] =
      limits_[variable].first + (static_cast<double>(quotient) * (limits_[variable].second - limits_[variable].first) /
                                 static_cast<double>(number_of_intervals_[variable]));
  definition = definition % divider;
}

std::vector<double> FrolovaSMultIntTrapezSEQ::GetPointFromNumber(unsigned int number) {
  std::vector<double> point(limits_.size());
  unsigned int definition = number;
  Recursive(point, definition, 1, static_cast<unsigned int>(limits_.size()) - 1);

  return point;
}

bool FrolovaSMultIntTrapezSEQ::ValidationImpl() {
  // std::cout << "[SEQ VALIDATION] Starting validation" << std::endl;

  auto input = GetInput();

  // std::cout << "[SEQ VALIDATION] Checking input:" << std::endl;
  // std::cout << "[SEQ VALIDATION]   limits.empty(): " << input.limits.empty() << std::endl;
  // std::cout << "[SEQ VALIDATION]   number_of_intervals.empty(): " << input.number_of_intervals.empty() << std::endl;

  if (input.limits.empty() || input.number_of_intervals.empty()) {
    // std::cout << "[SEQ VALIDATION] FAILED - empty limits or intervals" << std::endl;
    return false;
  }

  // std::cout << "[SEQ VALIDATION]   limits.size(): " << input.limits.size() << std::endl;
  // std::cout << "[SEQ VALIDATION]   intervals.size(): " << input.number_of_intervals.size() << std::endl;

  if (input.limits.size() != input.number_of_intervals.size()) {
    // std::cout << "[SEQ VALIDATION] FAILED - sizes don't match" << std::endl;
    return false;
  }

  // std::cout << "[SEQ VALIDATION]   function pointer: " << (input.function ? "NOT null" : "NULL") << std::endl;

  if (!input.function) {
    // std::cout << "[SEQ VALIDATION] FAILED - function is null" << std::endl;
    return false;
  }

  // Check each limit
  for (size_t i = 0; i < input.limits.size(); i++) {
    // std::cout << "[SEQ VALIDATION]   Limit " << i << ": [" << input.limits[i].first << ", " << input.limits[i].second
    //           << "], intervals: " << input.number_of_intervals[i] << std::endl;

    if (input.limits[i].first >= input.limits[i].second) {
      // std::cout << "[SEQ VALIDATION] FAILED - limit " << i << " has first >= second" << std::endl;
      return false;
    }

    if (input.number_of_intervals[i] == 0) {
      // std::cout << "[SEQ VALIDATION] FAILED - interval " << i << " is zero" << std::endl;
      return false;
    }
  }

  // std::cout << "[SEQ VALIDATION] PASSED validation" << std::endl;
  return true;
}

bool FrolovaSMultIntTrapezSEQ::PreProcessingImpl() {
  // std::cout << "[SEQ PRE_PROCESSING] Starting pre-processing" << std::endl;

  auto input = GetInput();
  limits_ = input.limits;
  number_of_intervals_ = input.number_of_intervals;
  result_ = 0.0;

  // std::cout << "[SEQ PRE_PROCESSING] Initialized with limits size = " << limits_.size()
  //           << ", intervals size = " << number_of_intervals_.size() << std::endl;
  // std::cout << "[SEQ PRE_PROCESSING] Function is " << (input.function ? "NOT null" : "NULL") << std::endl;

  return true;
}

bool FrolovaSMultIntTrapezSEQ::RunImpl() {
  unsigned int count = 1;
  for (auto number_of_interval : number_of_intervals_) {
    count *= (number_of_interval + 1);
  }

  for (unsigned int i = 0; i < count; i++) {
    std::vector<double> point = GetPointFromNumber(i);
    result_ += static_cast<double>(CalculationOfCoefficient(point)) * GetInput().function(point);
  }

  for (unsigned int i = 0; i < limits_.size(); i++) {
    result_ *= (limits_[i].second - limits_[i].first) / static_cast<double>(number_of_intervals_[i]);
  }

  result_ /= std::pow(2.0, static_cast<double>(limits_.size()));

  return true;
}

bool FrolovaSMultIntTrapezSEQ::PostProcessingImpl() {
  GetOutput() = result_;

  return true;
}

}  // namespace frolova_s_mult_int_trapez
