#pragma once

#include "dolov_v_qsort_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dolov_v_qsort_batcher {

class DolovVQsortBatcherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit DolovVQsortBatcherMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace dolov_v_qsort_batcher
