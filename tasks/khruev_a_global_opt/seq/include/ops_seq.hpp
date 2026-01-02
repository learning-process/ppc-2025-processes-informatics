#pragma once

#include <vector>

#include "khruev_a_global_opt/common/include/common.hpp"
#include "task/include/task.hpp"

namespace khruev_a_global_opt {

class KhruevAGlobalOptSEQ : public BaseTask {
 public:
  explicit KhruevAGlobalOptSEQ(const InType &in);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Trial> trials_;
  OutType result_;

  double CalculateFunction(double t);
  void AddTrial(double t, double z);
};

}  // namespace khruev_a_global_opt
