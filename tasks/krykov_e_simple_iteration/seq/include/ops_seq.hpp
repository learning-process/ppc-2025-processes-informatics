#pragma once

#include "krykov_e_simple_iteration/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krykov_e_simple_iteration {

class KrykovESimpleIterationSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KrykovESimpleIterationSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace krykov_e_simple_iteration
