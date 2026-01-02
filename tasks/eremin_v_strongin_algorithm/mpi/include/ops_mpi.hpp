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
};

}  // namespace eremin_v_strongin_algorithm
