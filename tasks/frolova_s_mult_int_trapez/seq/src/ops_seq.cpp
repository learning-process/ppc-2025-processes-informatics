#include "frolova_s_mult_int_trapez/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "frolova_s_mult_int_trapez/common/include/common.hpp"

namespace frolova_s_mult_int_trapez {

FrolovaSMultIntTrapezSEQ::FrolovaSMultIntTrapezSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

unsigned int FrolovaSMultIntTrapezSEQ::CalculationOfCoefficient(const std::vector<double> &point) {
  unsigned int degree = limits.size();
  for (unsigned int i = 0; i < limits.size(); i++) {
    if ((limits[i].first == point[i]) || (limits[i].second == point[i])) {
      degree--;
    }
  }

  return pow(2, degree);
}

void FrolovaSMultIntTrapezSEQ::Recursive(std::vector<double> &_point, unsigned int &definition, unsigned int divider,
                                         unsigned int variable) {
  if (variable > 0) {
    Recursive(_point, definition, divider * (number_of_intervals[variable] + 1), variable - 1);
  }
  _point[variable] = limits[variable].first + definition / divider *
                                                  (limits[variable].second - limits[variable].first) /
                                                  number_of_intervals[variable];
  definition = definition % divider;
}

std::vector<double> FrolovaSMultIntTrapezSEQ::GetPointFromNumber(unsigned int number) {
  std::vector<double> point(limits.size());
  unsigned int definition = number;
  Recursive(point, definition, 1, limits.size() - 1);

  return point;
}

bool FrolovaSMultIntTrapezSEQ::ValidationImpl() {
  return true;
}

bool FrolovaSMultIntTrapezSEQ::PreProcessingImpl() {
  limits = GetInput().limits;
  number_of_intervals = GetInput().number_of_intervals;
  result = 0.0;

  return (limits.size() == number_of_intervals.size()) && (limits.size() > 0);
}

bool FrolovaSMultIntTrapezSEQ::RunImpl() {
  unsigned int count = 1;
  for (unsigned int i = 0; i < number_of_intervals.size(); i++) {
    count *= (number_of_intervals[i] + 1);
  }

  for (unsigned int i = 0; i < count; i++) {
    std::vector<double> point = GetPointFromNumber(i);
    result += CalculationOfCoefficient(point) * GetInput().function(point);
  }

  for (unsigned int i = 0; i < limits.size(); i++) {
    result *= (limits[i].second - limits[i].first) / number_of_intervals[i];
  }

  result /= pow(2, limits.size());

  return true;
}

bool FrolovaSMultIntTrapezSEQ::PostProcessingImpl() {
  GetOutput() = result;

  return true;
}

}  // namespace frolova_s_mult_int_trapez
