#pragma once

#include "task/include/task.hpp"
#include "volkov_a_count_word_line/common/include/common.hpp"

namespace volkov_a_count_word_line {

class VolkovACountWordLineSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VolkovACountWordLineSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace volkov_a_count_word_line
