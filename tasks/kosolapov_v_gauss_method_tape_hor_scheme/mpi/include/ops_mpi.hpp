#pragma once

#include "kosolapov_v_gauss_method_tape_hor_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kosolapov_v_gauss_method_tape_hor_scheme {

class KosolapovVGaussMethodTapeHorSchemeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KosolapovVGaussMethodTapeHorSchemeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void DistributeDataFromRoot(std::vector<std::vector<double>> &local_matrix, std::vector<double> &local_rsd,
                              std::vector<int> &local_row_indices, int start, int local_rows, int columns,
                              int processes_count, int rows_per_proc, int remainder);
};

}  // namespace kosolapov_v_gauss_method_tape_hor_scheme
