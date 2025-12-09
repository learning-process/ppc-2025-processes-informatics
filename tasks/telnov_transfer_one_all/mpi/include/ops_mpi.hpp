#pragma once

#include "telnov_transfer_one_all/common/include/common.hpp"
#include "task/include/task.hpp"

namespace telnov_transfer_one_all {

class TelnovTransferOneAllMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TelnovTransferOneAllMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace telnov_transfer_one_all
