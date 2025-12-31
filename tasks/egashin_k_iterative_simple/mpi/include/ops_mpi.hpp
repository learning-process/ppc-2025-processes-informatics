#pragma once

#include <vector>

#include "egashin_k_iterative_simple/common/include/common.hpp"
#include "task/include/task.hpp"

namespace egashin_k_iterative_simple {

class EgashinKIterativeSimpleMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit EgashinKIterativeSimpleMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void CalculateDistribution(int size, int n, std::vector<int> &counts, std::vector<int> &displs);
  static double CalculateTau(const std::vector<std::vector<double>> &matrix, int start_row, int end_row);
  static double CalculateNorm(const std::vector<double> &v);
  static bool CheckConvergence(const std::vector<double> &x_old, const std::vector<double> &x_new, double tol);
  void BroadcastInputData(int n, std::vector<std::vector<double>> &a_local, std::vector<double> &b,
                          std::vector<double> &x, double &tolerance, int &max_iterations);
  static void PerformIteration(const std::vector<std::vector<double>> &a_local, const std::vector<double> &b,
                               const std::vector<double> &x, std::vector<double> &x_new, int start_row,
                               int local_rows, int n, double tau, const std::vector<int> &counts,
                               const std::vector<int> &displs);
};

}  // namespace egashin_k_iterative_simple
