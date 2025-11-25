#pragma once

#include "kondakov_v_min_val_in_matrix_str/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kondakov_v_min_val_in_matrix_str {

class KondakovVMinValMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KondakovVMinValMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void BroadcastMatrixMetadata(const InType &in_data, size_t &total_rows, size_t &cols);
  void HandleEmptyMatrix(int rank);
  static void ComputeRowRanges(int n, int rank, size_t total_rows, int &st_row, int &en_row);
  static void FindLocalMinima(const InType &in_data, int st_row, int en_row, size_t cols, std::vector<int> &temp);
  static void PrepareGathervParams(int n, size_t total_rows, int extra, std::vector<int> &recv_counts,
                                   std::vector<int> &displs);
};
}  // namespace kondakov_v_min_val_in_matrix_str
