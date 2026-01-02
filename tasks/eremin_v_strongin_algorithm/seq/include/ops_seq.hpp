#pragma once

#include "eremin_v_strongin_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace eremin_v_strongin_algorithm {

class EreminVStronginAlgorithmSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit EreminVStronginAlgorithmSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  double lower_bound = 0.0;
  double upper_bound = 0.0;
  double epsilon = 0.0;
  int max_iterations = 0;

  std::function<double(double)> objective_function;

  std::vector<double> search_points{};
  std::vector<double> function_values{};
};

}  // namespace eremin_v_strongin_algorithm
