#pragma once

#include <mpi.h>

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "frolova_s_mult_int_trapez/common/include/common.hpp"
#include "task/include/task.hpp"

namespace frolova_s_mult_int_trapez {

class FrolovaSMultIntTrapezMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit FrolovaSMultIntTrapezMPI(const InType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  unsigned int CalculationOfCoefficient(const std::vector<double> &point);
  void Recursive(std::vector<double> &_point, unsigned int &definition, unsigned int divider, unsigned int variable);
  std::vector<double> GetPointFromNumber(unsigned int number);

  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> number_of_intervals;
  double result;
};

}  // namespace frolova_s_mult_int_trapez
