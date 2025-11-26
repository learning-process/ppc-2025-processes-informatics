#pragma once

#include "nikitin_a_vec_sign_rotation/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikitin_a_vec_sign_rotation {

class NikitinAVecSignRotationMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit NikitinAVecSignRotationMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace nikitin_a_vec_sign_rotation
