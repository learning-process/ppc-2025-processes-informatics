#pragma once

#include <vector>

#include "shvetsova_k_rad_sort_batch_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shvetsova_k_rad_sort_batch_merge {

class ShvetsovaKRadSortBatchMergeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit ShvetsovaKRadSortBatchMergeMPI(const InType &in);

 private:
  std::vector<double> data_;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // вспомогательные функции //
  static void CreateDistribution(int proc_count, int size, std::vector<int> &counts, std::vector<int> &displs);

  static void ScatterData(const std::vector<double> &data, std::vector<double> &local, const std::vector<int> &counts,
                          const std::vector<int> &displs, int rank);

  static void BatcherOddEvenMergeParallel(std::vector<double> &local, const std::vector<int> &counts, int rank,
                                          int proc_count);

  static void GatherAndBroadcast(std::vector<double> &data, const std::vector<double> &local,
                                 const std::vector<int> &counts, const std::vector<int> &displs, int rank);

  static void RadixSortLocal(std::vector<double> &vec);

  // Batcher's merge вспомогательные функции
  static void BatcherOddEvenMergeSequential(std::vector<double> &arr1, std::vector<double> &arr2,
                                            std::vector<double> &result);

  static void ExchangeAndMerge(std::vector<double> &local, const std::vector<int> &counts, int partner,
                               bool keep_smaller);
};

}  // namespace shvetsova_k_rad_sort_batch_merge
