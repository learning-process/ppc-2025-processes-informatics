#pragma once

#include "votincev_d_alternating_values/common/include/common.hpp"
#include "task/include/task.hpp"

namespace votincev_d_alternating_values {

class VotincevDAlternatingValuesSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VotincevDAlternatingValuesSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace votincev_d_alternating_values
