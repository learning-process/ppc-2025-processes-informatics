#pragma once

#include "eremin_v_strongin_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace eremin_v_strongin_algorithm {

class EreminVStronginAlgorithmMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit EreminVStronginAlgorithmMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  double CalculateLipschitzEstimate(int rank, int size, const std::vector<double> &search_points,
                                    const std::vector<double> &function_values);
};

}  // namespace eremin_v_strongin_algorithm
