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

  std::vector<double> x0;
  std::vector<std::vector<double>> A;
  std::vector<double> b;
  double eps;
  size_t iterations;
};

}  // namespace romanova_v_jacobi_method_processes
