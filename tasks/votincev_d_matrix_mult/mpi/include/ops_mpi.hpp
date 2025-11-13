#pragma once

#include <array>
#include <vector>

#include "task/include/task.hpp"
#include "votincev_d_matrix_mult/common/include/common.hpp"

namespace votincev_d_matrix_mult {

class VotincevDMatrixMultMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VotincevDMatrixMultMPI(const InType &in);

 private:
  int m_;
  int n_;
  int k_;
  std::vector<double> A_;
  std::vector<double> B_;
  std::vector<double> result_;
  void MultiplyBlock(int start_row, int end_row, std::vector<double> &out);
  void SyncResults();

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace votincev_d_matrix_mult
