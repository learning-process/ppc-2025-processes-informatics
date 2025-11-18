#pragma once

#include <vector>

#include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"
#include "task/include/task.hpp"

namespace leonova_a_most_diff_neigh_vec_elems {

class LeonovaAMostDiffNeighVecElemsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LeonovaAMostDiffNeighVecElemsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void ProcessWithMultipleProcesses(int rank, int size, int total_size, const std::vector<int> &input_vec);
  void ProcessLocalData(int rank, int actual_processes, int total_size, const std::vector<int> &input_vec,
                        int &local_max_diff, int &local_first, int &local_second);
  void ReceiveLocalData(int rank, int actual_processes, const std::vector<int> &input_vec, int my_size,
                        std::vector<int> &local_data, int total_size);
  static void SendDataToProcess(int dest, int actual_processes, const std::vector<int> &input_vec,
                                int total_size);  // Добавлен static
  static void FindLocalMaxDiff(const std::vector<int> &local_data, int &local_max_diff, int &local_first,
                               int &local_second);  // Добавлен static
  void GatherAndProcessResults(int rank, int actual_processes, int local_max_diff, int local_first, int local_second,
                               int size);
  void FindGlobalMaxDiff(const std::vector<int> &all_diffs, const std::vector<int> &all_firsts,
                         const std::vector<int> &all_seconds, int actual_processes);
  void BroadcastResult(int rank);
  void ProcessSequentially(const std::vector<int> &input_vec);
};

}  // namespace leonova_a_most_diff_neigh_vec_elems
