#pragma once

#include "kosolapov_v_gauss_method_tape_hor_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kosolapov_v_gauss_method_tape_hor_scheme {

class KosolapovVGaussMethodTapeHorSchemeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KosolapovVGaussMethodTapeHorSchemeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kosolapov_v_gauss_method_tape_hor_scheme
