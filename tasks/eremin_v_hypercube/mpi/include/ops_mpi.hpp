#pragma once

#include "eremin_v_hypercube/common/include/common.hpp"
#include "task/include/task.hpp"

namespace eremin_v_hypercube {

class EreminVHypercubeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit EreminVHypercubeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace eremin_v_hypercube
