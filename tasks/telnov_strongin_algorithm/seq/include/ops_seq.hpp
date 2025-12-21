#pragma once

#include "task/include/task.hpp"
#include "telnov_strongin_algorithm/common/include/common.hpp"

namespace telnov_strongin_algorithm {

class TelnovStronginAlgorithmSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TelnovStronginAlgorithmSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace telnov_strongin_algorithm
