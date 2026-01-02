#pragma once

#include <cstdint>
#include <vector>

#include "task/include/task.hpp"
#include "volkov_a_radix_batcher/common/include/common.hpp"

namespace volkov_a_radix_batcher {

class VolkovARadixBatcherSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VolkovARadixBatcherSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void RadixSortDouble(std::vector<double> &data);
  static uint64_t DoubleToOrderedInt(double d);
  static double OrderedIntToDouble(uint64_t k);
};

}  // namespace volkov_a_radix_batcher