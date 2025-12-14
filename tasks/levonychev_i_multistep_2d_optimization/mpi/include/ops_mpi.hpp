#pragma once

#include "levonychev_i_multistep_2d_optimization/common/include/common.hpp"
#include "task/include/task.hpp"

namespace levonychev_i_multistep_2d_optimization {

class LevonychevIMultistep2dOptimizationMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LevonychevIMultistep2dOptimizationMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace levonychev_i_multistep_2d_optimization
