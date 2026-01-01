#include "frolova_s_mult_int_trapez/seq/include/ops_seq.hpp"

#include <cmath>
#include <vector>

#include "frolova_s_mult_int_trapez/common/include/common.hpp"

namespace frolova_s_mult_int_trapez {

FrolovaSMultIntTrapezSEQ::FrolovaSMultIntTrapezSEQ(const InType &in)
    : result_(0.0), limits_(in.limits), number_of_intervals_(in.number_of_intervals) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
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
  return true;
}

bool FrolovaSMultIntTrapezSEQ::PreProcessingImpl() {
  limits_ = GetInput().limits;
  number_of_intervals_ = GetInput().number_of_intervals;
  result_ = 0.0;

  return (limits_.size() == number_of_intervals_.size()) && !limits_.empty();
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
