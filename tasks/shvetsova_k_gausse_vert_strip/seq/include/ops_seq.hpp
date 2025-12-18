#pragma once

#include <vector>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shvetsova_k_gausse_vert_strip {

class ShvetsovaKGaussVertStripSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ShvetsovaKGaussVertStripSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int sizeOfRib = 0;
};

}  // namespace shvetsova_k_gausse_vert_strip
