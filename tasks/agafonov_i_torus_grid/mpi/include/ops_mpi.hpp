#pragma once

#include <gtest/gtest.h>

#include "agafonov_i_torus_grid/common/include/common.hpp"
#include "task/include/task.hpp"

namespace agafonov_i_torus_grid {

class TorusGridTaskMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TorusGridTaskMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace agafonov_i_torus_grid
