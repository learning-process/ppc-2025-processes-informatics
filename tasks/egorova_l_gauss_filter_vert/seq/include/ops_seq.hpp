#pragma once

#include "egorova_l_gauss_filter_vert/common/include/common.hpp"
#include "task/include/task.hpp"

namespace egorova_l_gauss_filter_vert {

class EgorovaLGaussFilterVertSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit EgorovaLGaussFilterVertSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace egorova_l_gauss_filter_vert
