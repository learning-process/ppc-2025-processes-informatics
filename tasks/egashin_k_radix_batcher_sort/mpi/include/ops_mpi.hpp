#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "egashin_k_radix_batcher_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace egashin_k_radix_batcher_sort {

class EgashinKRadixBatcherSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit EgashinKRadixBatcherSortMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void RadixSort(std::vector<double> &arr);
  static uint64_t DoubleToSortable(double value);
  static double SortableToDouble(uint64_t bits);
  static void BatcherOddEvenMerge(std::vector<double> &arr, int lo, int n, int r);
  static void BatcherOddEvenMergeSort(std::vector<double> &arr, int lo, int n);
  static void CompareExchange(std::vector<double> &arr, int i, int j);
  static std::vector<std::pair<int, int>> GenerateBatcherNetwork(int n);
  static void MergeWithPartner(std::vector<double> &local_data, int partner_rank, int rank, bool keep_lower);
  static void DistributeData(int total_size, int world_size, int rank, std::vector<double> &data,
                             std::vector<int> &counts, std::vector<int> &displs, std::vector<double> &local_data);
  static void PerformBatcherMerge(std::vector<double> &local_data, int world_size, int rank);
  static void GatherResults(std::vector<double> &local_data, int total_size, int world_size, int rank,
                            std::vector<double> &sorted_data);
};

}  // namespace egashin_k_radix_batcher_sort
