#pragma once

#include "task/include/task.hpp"
#include "telnov_transfer_one_all/common/include/common.hpp"

namespace telnov_transfer_one_all {

template <typename T>
class TelnovTransferOneAllSEQ : public BaseTask<T> {
 public:
  using InType = std::vector<T>;
  using OutType = InType;
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TelnovTransferOneAllSEQ(const std::vector<T>& input) : BaseTask<T>(input) {}

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace telnov_transfer_one_all
