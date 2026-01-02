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

  // ===== ДОП ФУНКЦИИ =====
  static void CreateDistribution(int proc_count, int size, std::vector<int> &counts, std::vector<int> &displs,
                                 int rank);

  static void RadixSortLocal(std::vector<double> &vec);

  static void OddEvenMergeStep(std::vector<double> &local, int rank, int proc_count);
};

}  // namespace shvetsova_k_rad_sort_batch_merge
