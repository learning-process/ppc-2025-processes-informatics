#pragma once

#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sizov_d_global_search {

class SizovDGlobalSearchMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SizovDGlobalSearchMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> points_;
  std::vector<double> values_;
  double best_point_ = 0.0;
  double best_value_ = 0.0;
  int iterations_ = 0;
  bool converged_ = false;
};

}  // namespace sizov_d_global_search
