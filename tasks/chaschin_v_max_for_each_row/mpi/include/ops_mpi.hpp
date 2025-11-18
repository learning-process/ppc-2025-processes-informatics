#pragma once

#include "chaschin_v_max_for_each_row/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chaschin_v_max_for_each_row {

class ChaschinVMaxForEachRow : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ChaschinVMaxForEachRow(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  struct RowRange {
    int start;
    int count;
  };

  RowRange ComputeRange(int nrows, int rank, int size);
  std::vector<std::vector<float>> DistributeRows(const std::vector<std::vector<float>> &mat, int rank, int size,
                                                 const RowRange &range);
  std::vector<float> ComputeLocalMax(const std::vector<std::vector<float>> &local_mat);

  void GatherResults(std::vector<float> &out, const std::vector<float> &local_out, int rank, int size,
                     const RowRange &range);
};

}  // namespace chaschin_v_max_for_each_row
