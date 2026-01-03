#pragma once

#include <vector>

#include "shvetsova_k_rad_sort_batch_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shvetsova_k_rad_sort_batch_merge {

class ShvetsovaKRadSortBatchMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit ShvetsovaKRadSortBatchMergeSEQ(const InType &in);

 private:
  std::vector<double> data_;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // доп функции //
  static void RadixSort(std::vector<double> &vec);
  static void BatcherOddEvenMergeSort(std::vector<double> &vec, int left, int right);
  static void ExecuteBatcherStep(std::vector<double> &vec, int left, int n, int p, int k);
  static void CompareAndSwap(std::vector<double> &vec, int i, int j);
};

}  // namespace shvetsova_k_rad_sort_batch_merge
