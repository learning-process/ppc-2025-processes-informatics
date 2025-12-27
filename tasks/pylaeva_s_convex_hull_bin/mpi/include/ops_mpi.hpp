#pragma once

#include "pylaeva_s_convex_hull_bin/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pylaeva_s_convex_hull_bin {

class PylaevaSConvexHullBinMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PylaevaSConvexHullBinMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace pylaeva_s_convex_hull_bin
