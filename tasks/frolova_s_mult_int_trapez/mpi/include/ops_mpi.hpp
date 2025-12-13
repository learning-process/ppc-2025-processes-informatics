#pragma once

#include <mpi.h>

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "frolova_s_star_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace frolova_s_star_topology {

class FrolovaSStarTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit FrolovaSStarTopologyMPI(const InType &in);

 private:
  unsigned int CalculationOfCoefficient(const std::vector<double> &point);
  void Recursive(std::vector<double> &_point, unsigned int &definition, unsigned int divider, unsigned int variable);
  std::vector<double> GetPointFromNumber(unsigned int number);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> number_of_intervals;
  double result;
};

}  // namespace frolova_s_star_topology
