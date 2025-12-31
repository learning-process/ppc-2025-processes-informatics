#pragma once

#include <vector>

#include "dolov_v_qsort_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dolov_v_qsort_batcher {

class DolovVQsortBatcherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit DolovVQsortBatcherMPI(const InType &in);
  ~DolovVQsortBatcherMPI() override = default;

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Оставляем только то, что реально нужно для текущего алгоритма
  void SetupWorkload(int total_size, int proc_count, std::vector<int> &counts, std::vector<int> &offsets);
  void RunBatcherMerge(int rank, int proc_count, std::vector<double> &local_vec);
  void ExchangeAndMerge(int partner, std::vector<double> &local_vec, bool keep_smaller);

  std::vector<double> local_data_;
  std::vector<int> send_counts_;
  std::vector<int> displacements_;
  int total_elements_ = 0;
};

}  // namespace dolov_v_qsort_batcher
