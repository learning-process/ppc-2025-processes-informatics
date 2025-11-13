#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "votincev_d_matrix_mult/common/include/common.hpp"

namespace votincev_d_matrix_mult {

class VotincevDMatrixMultSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VotincevDMatrixMultSEQ(const InType &in);

 private:
  int m_;
  int n_;
  int k_;
  std::vector<double> A_;
  std::vector<double> B_;
  std::vector<double> result_;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace votincev_d_matrix_mult
