#pragma once

#include <mpi.h>

#include <vector>

#include "goriacheva_k_strassen_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace goriacheva_k_Strassen_algorithm {

class GoriachevaKStrassenAlgorithmMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit GoriachevaKStrassenAlgorithmMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType input_matrices_;
  OutType result_matrix_;

  Matrix mpi_Strassen_top(const Matrix &A, const Matrix &B);
};

}  // namespace goriacheva_k_Strassen_algorithm
