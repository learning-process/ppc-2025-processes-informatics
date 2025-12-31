#pragma once

#include <cstdint>
#include <vector>

#include "egashin_k_radix_batcher_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace egashin_k_radix_batcher_sort {

class EgashinKRadixBatcherSortSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit EgashinKRadixBatcherSortSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void RadixSort(std::vector<double> &arr);
  static uint64_t DoubleToSortable(double value);
  static double SortableToDouble(uint64_t bits);
};

}  // namespace egashin_k_radix_batcher_sort
