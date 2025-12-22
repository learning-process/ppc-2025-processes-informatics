#pragma once

#include "romanova_v_jacobi_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace romanova_v_jacobi_method {

class RomanovaVJacobiMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RomanovaVJacobiMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> x_;
  std::vector<std::vector<double>> A_;
  std::vector<double> b_;
  double eps_;
  size_t maxIterations_;
};

}  // namespace romanova_v_jacobi_method
