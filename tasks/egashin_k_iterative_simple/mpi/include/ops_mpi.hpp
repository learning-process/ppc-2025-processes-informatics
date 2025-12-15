#pragma once

#include <vector>

#include "egashin_k_iterative_simple/common/include/common.hpp"
#include "task/include/task.hpp"

namespace egashin_k_iterative_simple {

class TestTaskMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit TestTaskMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void CalculateDistribution(int size, int n, std::vector<int> &counts, std::vector<int> &displs);
  static double CalculateTau(const std::vector<std::vector<double>> &A, int start_row, int end_row);
  static double CalculateNorm(const std::vector<double> &v);
  static bool CheckConvergence(const std::vector<double> &x_old, const std::vector<double> &x_new, double tolerance);
};

}  // namespace egashin_k_iterative_simple

