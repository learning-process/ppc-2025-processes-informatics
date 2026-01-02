#pragma once

#include <cstdint>
#include <vector>

#include "task/include/task.hpp"
#include "volkov_a_radix_batcher/common/include/common.hpp"

namespace volkov_a_radix_batcher {

class VolkovARadixBatcherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VolkovARadixBatcherMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void RadixSortDouble(std::vector<double> &data);
  static uint64_t DoubleToOrderedInt(double d);
  static double OrderedIntToDouble(uint64_t k);

  static void CalculateDistribution(int world_size, int total_elements, std::vector<int> &counts,
                                    std::vector<int> &displs);

  static void ParallelMergeSort(int rank, int world_size, const std::vector<int> &counts,
                                std::vector<double> &local_vec);

  static void ExchangeAndMerge(int rank, int neighbor, const std::vector<int> &counts, std::vector<double> &local_vec,
                               std::vector<double> &buffer_recv, std::vector<double> &buffer_merge);
};

}  // namespace volkov_a_radix_batcher