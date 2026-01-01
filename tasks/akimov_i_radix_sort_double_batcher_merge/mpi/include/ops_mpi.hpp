#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "akimov_i_radix_sort_double_batcher_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace akimov_i_radix_sort_double_batcher_merge {

class AkimovIRadixBatcherSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit AkimovIRadixBatcherSortMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Radix helpers
  static uint64_t packDouble(double v) noexcept;
  static double unpackDouble(uint64_t k) noexcept;
  static void lsdRadixSort(std::vector<double> &arr);

  // Batcher / merge-network helpers
  static void cmpSwap(std::vector<double> &arr, int i, int j) noexcept;
  static void oddEvenMergeRec(std::vector<double> &arr, int start, int len, int stride);
  static void oddEvenMergeSortRec(std::vector<double> &arr, int start, int len);

  static std::vector<std::pair<int, int>> buildOddEvenPhasePairs(int procs);
  static void exchangeAndSelect(std::vector<double> &local, int partner, int rank, bool keep_lower);

  // MPI data distribution helpers
  static void computeCountsDispls(int total, int world, std::vector<int> &counts, std::vector<int> &displs);
  static void scatterData(int total, int world, int rank, std::vector<double> &data, std::vector<int> &counts,
                          std::vector<int> &displs, std::vector<double> &local);
  static void gatherData(int total, int world, int rank, std::vector<double> &local, std::vector<double> &out);
  static void performNetworkMerge(std::vector<double> &local, int world, int rank);
};

}  // namespace akimov_i_radix_sort_double_batcher_merge
