#pragma once

#include "leonova_a_star/common/include/common.hpp"
#include "task/include/task.hpp"

namespace leonova_a_star {

class LeonovaAStarSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LeonovaAStarSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace leonova_a_star
