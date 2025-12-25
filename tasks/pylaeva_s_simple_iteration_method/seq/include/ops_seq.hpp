#pragma once

#include <cstddef>  // для size_t
#include <vector>

#include "pylaeva_s_simple_iteration_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pylaeva_s_simple_iteration_method {

class PylaevaSSimpleIterationMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PylaevaSSimpleIterationMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  const double kEps = 1e-6;
  const int kMaxIterations = 10000;

  bool NotNullDeterm(const std::vector<double> &a, size_t n);
  bool DiagonalDominance(const std::vector<double> &a, size_t n);
};

}  // namespace pylaeva_s_simple_iteration_method
