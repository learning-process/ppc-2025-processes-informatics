#pragma once

#include "ashihmin_d_calculate_integrals_by_simpson/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ashihmin_d_calculate_integrals_by_simpson {

class AshihminDCalculateIntegralsBySimpsonMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit AshihminDCalculateIntegralsBySimpsonMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  
  double IntegrandFunction(const std::vector<double>& coords);
  void RecursiveIntegration(int dim, std::vector<double>& coords, 
                           const std::vector<double>& h, double& sum,
                           int start_i, int end_i);
};

}  // namespace ashihmin_d_calculate_integrals_by_simpson