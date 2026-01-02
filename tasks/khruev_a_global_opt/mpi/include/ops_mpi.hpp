#pragma once

#include <vector>

#include "khruev_a_global_opt/common/include/common.hpp"

namespace khruev_a_global_opt {

class KhruevAGlobalOptMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit KhruevAGlobalOptMPI(const InType &in);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Trial> trials_;
  OutType result_;

  double CalculateFunction(double t);
  void AddTrialUnsorted(double t, double z);
};

}  // namespace khruev_a_global_opt
