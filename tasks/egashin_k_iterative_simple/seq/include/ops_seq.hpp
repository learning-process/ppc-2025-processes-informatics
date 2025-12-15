#pragma once

#include "egashin_k_iterative_simple/common/include/common.hpp"
#include "task/include/task.hpp"

namespace egashin_k_iterative_simple {

class TestTaskSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TestTaskSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  double CalculateTau(const std::vector<std::vector<double>> &A);
  double CalculateNorm(const std::vector<double> &v);
  bool CheckConvergence(const std::vector<double> &x_old, const std::vector<double> &x_new, double tolerance);
};

}  // namespace egashin_k_iterative_simple
