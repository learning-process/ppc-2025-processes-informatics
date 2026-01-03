#pragma once

#include "dorofeev_i_ccs_matrix_production/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dorofeev_i_ccs_matrix_production {

class DorofeevICCSMatrixProductionMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit DorofeevICCSMatrixProductionMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace dorofeev_i_ccs_matrix_production
