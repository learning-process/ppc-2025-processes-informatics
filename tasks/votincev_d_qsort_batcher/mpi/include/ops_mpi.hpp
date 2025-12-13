#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "votincev_d_qsort_batcher/common/include/common.hpp"

namespace votincev_d_qsort_batcher {

class VotincevDQsortBatcherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VotincevDQsortBatcherMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // ==============================
  // мои дополнительные функции ===
  void QuickSort(double *arr, int left, int right);
};

}  // namespace votincev_d_qsort_batcher
