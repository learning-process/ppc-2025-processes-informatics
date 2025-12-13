#pragma once

#include <vector>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace frolova_s_sum_elem_matrix {

class FrolovaSSumElemMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit FrolovaSSumElemMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void BroadcastMetadata(int &rows);
  static void BroadcastRowSizes(int rows, std::vector<int> &row_sizes);
  static void FlattenMatrixOnRoot(int rank, const std::vector<std::vector<int>> &matrix,
                                  const std::vector<int> &row_sizes, std::vector<int> &flat_data,
                                  std::vector<int> &displs);
  static void ComputeDistribution(int size, int rows, std::vector<int> &counts, std::vector<int> &displacements);
  static void ComputeSendCounts(int rank, int size, const std::vector<int> &counts,
                                const std::vector<int> &displacements, const std::vector<int> &row_sizes,
                                std::vector<int> &sendcounts, std::vector<int> &senddispls);
};

}  // namespace frolova_s_sum_elem_matrix
