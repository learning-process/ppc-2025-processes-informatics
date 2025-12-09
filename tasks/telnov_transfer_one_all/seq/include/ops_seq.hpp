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
  explicit TelnovTransferOneAllSEQ(const InType &in);
  explicit TelnovTransferOneAllSEQ(const TelnovTransferOneAllMPI<T> &mpi) 
    : TelnovTransferOneAllSEQ(mpi.getData()) {}

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace telnov_transfer_one_all
