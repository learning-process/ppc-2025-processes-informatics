#pragma once

#include "../../common/include/common.hpp"
#include "task/include/task.hpp"

namespace shekhirev_v_cg_method_seq {

class ConjugateGradientSeq : public shekhirev_v_cg_method::BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ConjugateGradientSeq(const shekhirev_v_cg_method::InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shekhirev_v_cg_method_seq
