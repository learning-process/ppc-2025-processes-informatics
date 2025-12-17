#pragma once

#include "task/include/task.hpp"
#include "shkrebko_m_shell_sort_batcher_merge/common/include/common.hpp"

namespace shkrebko_m_shell_sort_batcher_merge {

class ShkrebkoMShellSortBatcherMergeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ShkrebkoMShellSortBatcherMergeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static void ShellSort(std::vector<int> &arr);
  static void BatcherMerge(std::vector<int> &local_data, int rank, int world_size, int total_size);
  
  // Вспомогательные функции (можно сделать private или оставить как static)
  static int CalculateLocalSize(int total_size, int rank, int world_size);
  static std::vector<int> CalculateInterval(int total_size, int rank, int world_size);
  static void MergeAndSplit(std::vector<int> &a, std::vector<int> &b, bool keep_smaller);
  static void ExchangeData(std::vector<int> &local_data, int rank, int neighbor_rank, 
                           int total_size, int world_size);
  static void EvenPhase(std::vector<int> &local_data, int rank, int world_size, int total_size);
  static void OddPhase(std::vector<int> &local_data, int rank, int world_size, int total_size);
};

}  // namespace shkrebko_m_shell_sort_batcher_merge