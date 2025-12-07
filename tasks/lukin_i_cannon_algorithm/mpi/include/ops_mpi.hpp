#pragma once

#include "lukin_i_cannon_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lukin_i_cannon_algorithm {

class LukinICannonAlgorithmMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LukinICannonAlgorithmMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void MulNSum(double *a, double *b, double *c, int size);

  int size_ = 0;
};

}  // namespace lukin_i_cannon_algorithm
