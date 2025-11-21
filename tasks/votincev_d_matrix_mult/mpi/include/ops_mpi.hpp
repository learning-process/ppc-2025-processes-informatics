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

  static void MatrixPartMult(int k, int n,std::vector<double>& local_matrix,const std::vector<double>& matrix_B );
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace votincev_d_matrix_mult
