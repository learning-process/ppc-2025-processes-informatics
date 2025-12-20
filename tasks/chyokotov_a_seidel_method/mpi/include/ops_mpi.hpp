#pragma once

#include "chyokotov_a_seidel_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chyokotov_a_seidel_method {

class ChyokotovASeidelMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ChyokotovASeidelMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace chyokotov_a_seidel_method
