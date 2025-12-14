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

  // функции для разбиения RunImpl
  void ComputeDistribution(int proc_n, int total_size, std::vector<int> &sizes, std::vector<int> &offsets);
  void ScatterData(int rank, const std::vector<int> &sizes, const std::vector<int> &offsets,
                   std::vector<double> &local);
  void BatcherMergeSort(int rank, int proc_n, const std::vector<int> &sizes, std::vector<double> &local);
  void GatherResult(int rank, int total_size, const std::vector<int> &sizes, const std::vector<int> &offsets,
                    const std::vector<double> &local);
};

}  // namespace votincev_d_qsort_batcher
