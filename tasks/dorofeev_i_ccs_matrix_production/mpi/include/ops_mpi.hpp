#pragma once

#include "dorofeev_i_ccs_matrix_production/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dorofeev_i_ccs_matrix_production {

class DorofeevICCSMatrixProductionMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit DorofeevICCSMatrixProductionMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::pair<int, int> ComputeLocalColumnRange(int rank, int size, int total_cols);
  static std::vector<std::unordered_map<int, double>> ComputeLocalColumns(const InType &input, int start_col,
                                                                          int local_cols);
  static void GatherResultsOnRoot(OutType &output, std::vector<std::unordered_map<int, double>> &local_columns,
                                  int size, int rank);
  static void ProcessLocalResultsOnRoot(OutType &output, std::vector<std::unordered_map<int, double>> &local_columns,
                                        int &col_offset);
  static void ReceiveResultsFromProcess(OutType &output, int process_idx, int &col_offset);
  static void SendResultsToRoot(const std::vector<std::unordered_map<int, double>> &local_columns);
  static void BroadcastResults(OutType &output, int rank);
};

}  // namespace dorofeev_i_ccs_matrix_production
