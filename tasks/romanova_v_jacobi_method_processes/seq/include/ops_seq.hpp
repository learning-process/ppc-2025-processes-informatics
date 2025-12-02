#pragma once

#include "romanova_v_jacobi_method_processes/common/include/common.hpp"
#include "task/include/task.hpp"

namespace romanova_v_jacobi_method_processes {

class RomanovaVJacobiMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RomanovaVJacobiMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  bool isDiagonallyDominant(const std::vector<std::vector<double>>& matrix);
  bool isConverge(const std::vector<double>& prev, const std::vector<double>& curr);

  std::vector<double> x;
  std::vector<std::vector<double>> A;
  std::vector<double> b;
  double eps;
  size_t maxIterations;
  size_t size;
};

}  // namespace romanova_v_jacobi_method_processes
