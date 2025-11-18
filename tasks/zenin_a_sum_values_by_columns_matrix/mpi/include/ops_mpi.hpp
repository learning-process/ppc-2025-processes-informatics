#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"
#include "zenin_a_sum_values_by_columns_matrix/common/include/common.hpp"

namespace zenin_a_sum_values_by_columns_matrix {

class ZeninASumValuesByColumnsMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZeninASumValuesByColumnsMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void CalculateLocalSums(const std::vector<double> &matrix_data, size_t columns, size_t total_rows,
                                 size_t start_column, size_t cols_this_process, std::vector<double> &local_sums);
  static void PrepareGathervParameters(int world_size, size_t base_cols_per_process, size_t remain,
                                       std::vector<int> &recv_counts, std::vector<int> &displacements);
};

}  // namespace zenin_a_sum_values_by_columns_matrix
