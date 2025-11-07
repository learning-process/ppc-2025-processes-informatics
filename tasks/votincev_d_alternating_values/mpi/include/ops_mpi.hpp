#pragma once

#include <array>
#include <vector>

#include "task/include/task.hpp"
#include "votincev_d_alternating_values/common/include/common.hpp"

namespace votincev_d_alternating_values {

class VotincevDAlternatingValuesMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VotincevDAlternatingValuesMPI(const InType &in);

 private:
  std::vector<double> vect_data_;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int ProcessMaster(int process_n);
  void ProcessWorker();
  static void SendRangeToWorker(int start, int end, int worker);
  static std::array<int, 2> ReceiveRange();
  int CountSwaps(int start, int end);
  static int CollectWorkerResults(int process_n);
  static void SendResult(int result);
  void SyncResults(int &all_swaps);
};

}  // namespace votincev_d_alternating_values
