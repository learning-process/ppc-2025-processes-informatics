#pragma once

#include <cstddef>  // для size_t
#include <vector>

#include "pylaeva_s_simple_iteration_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pylaeva_s_simple_iteration_method {

class PylaevaSSimpleIterationMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PylaevaSSimpleIterationMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  const double kEps = 1e-6;
  const size_t kMaxIterations = 10000;

  bool NotNullDeterm(const std::vector<double> &a, size_t n);
  bool DiagonalDominance(const std::vector<double> &a, size_t n);
  void CalculateRowsDistribution(int proc_num, int n, std::vector<int> &row_counts_per_rank,
                                 std::vector<int> &row_offsets_per_rank);
};

}  // namespace pylaeva_s_simple_iteration_method
