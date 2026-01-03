#pragma once

#include "gusev_d_radix_double/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gusev_d_radix_double {

class GusevDRadixDoubleMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GusevDRadixDoubleMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace gusev_d_radix_double
