#pragma once

#include "dolov_v_qsort_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dolov_v_qsort_batcher {

class DolovVQsortBatcherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit DolovVQsortBatcherMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void applyQuicksort(double *array, int low, int high);
  int getHoarePartition(double *array, int low, int high);

  void setupWorkload(int totalSize, int procCount, std::vector<int> &counts, std::vector<int> &offsets);
  void splitData(const std::vector<double> &source, std::vector<double> &local, const std::vector<int> &counts,
                 const std::vector<int> &offsets);

  void runBatcherProcess(int rank, int procCount, std::vector<double> &localVec);
  void exchangeAndMerge(int rank, int partner, std::vector<double> &localVec, bool keepSmall);

  std::vector<double> mergeTwoSortedArrays(const std::vector<double> &first, const std::vector<double> &second);

  void collectData(std::vector<double> &globalRes, const std::vector<double> &local, int totalSize,
                   const std::vector<int> &counts, const std::vector<int> &offsets);
};

}  // namespace dolov_v_qsort_batcher
