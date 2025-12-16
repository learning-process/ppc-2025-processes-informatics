#pragma once

#include <vector>

#include "egashin_k_radix_batcher_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace egashin_k_radix_batcher_sort {

class TestTaskSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() { return ppc::task::TypeOfTask::kSEQ; }
  explicit TestTaskSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void RadixSort(std::vector<double> &arr);
  static uint64_t DoubleToSortable(double value);
  static double SortableToDouble(uint64_t value);
};

}  // namespace egashin_k_radix_batcher_sort

