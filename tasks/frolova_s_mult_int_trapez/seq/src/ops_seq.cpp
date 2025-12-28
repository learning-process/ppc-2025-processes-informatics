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
  unsigned int degree = limits_.size();
  for (unsigned int i = 0; i < limits_.size(); i++) {
    if ((limits_[i].first == point[i]) || (limits_[i].second == point[i])) {
      degree--;
    }
  }

  return pow(2, degree);
}

void FrolovaSMultIntTrapezSEQ::Recursive(std::vector<double> &_point, unsigned int &definition, unsigned int divider,
                                         unsigned int variable) {
  if (variable > 0) {
    Recursive(_point, definition, divider * (number_of_intervals_[variable] + 1), variable - 1);
  }
  _point[variable] = limits_[variable].first + definition / divider *
                                                   (limits_[variable].second - limits_[variable].first) /
                                                   number_of_intervals_[variable];
  definition = definition % divider;
}

std::vector<double> FrolovaSMultIntTrapezSEQ::GetPointFromNumber(unsigned int number) {
  std::vector<double> point(limits_.size());
  unsigned int definition = number;
  Recursive(point, definition, 1, limits_.size() - 1);

  return point;
}

bool FrolovaSMultIntTrapezSEQ::ValidationImpl() {
  return true;
}

bool FrolovaSMultIntTrapezSEQ::PreProcessingImpl() {
  limits_ = GetInput().limits;
  number_of_intervals_ = GetInput().number_of_intervals;
  result_ = 0.0;

  return (limits_.size() == number_of_intervals_.size()) && (limits_.size() > 0);
}

bool FrolovaSMultIntTrapezSEQ::RunImpl() {
  unsigned int count = 1;
  for (unsigned int i = 0; i < number_of_intervals_.size(); i++) {
    count *= (number_of_intervals_[i] + 1);
  }

  for (unsigned int i = 0; i < count; i++) {
    std::vector<double> point = GetPointFromNumber(i);
    result_ += CalculationOfCoefficient(point) * GetInput().function(point);
  }

  for (unsigned int i = 0; i < limits_.size(); i++) {
    result_ *= (limits_[i].second - limits_[i].first) / number_of_intervals_[i];
  }

  result_ /= pow(2, limits_.size());

  return true;
}

bool FrolovaSMultIntTrapezSEQ::PostProcessingImpl() {
  GetOutput() = result_;

  return true;
}

}  // namespace frolova_s_mult_int_trapez
