#pragma once

#include "ovsyannikov_n_star/common/include/common.hpp"
#include "task/include/task.hpp"

namespace ovsyannikov_n_star {

class OvsyannikovNStarMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit OvsyannikovNStarMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace ovsyannikov_n_star
