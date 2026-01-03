#pragma once

#include <vector>

#include "egashin_k_iterative_simple/common/include/common.hpp"
#include "task/include/task.hpp"

namespace egashin_k_iterative_simple {

class EgashinKIterativeSimpleSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit EgashinKIterativeSimpleSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static double CalculateTau(const std::vector<std::vector<double>> &matrix);
  static double CalculateNorm(const std::vector<double> &v);
  static bool CheckConvergence(const std::vector<double> &x_old, const std::vector<double> &x_new, double tol);
};

}  // namespace egashin_k_iterative_simple
