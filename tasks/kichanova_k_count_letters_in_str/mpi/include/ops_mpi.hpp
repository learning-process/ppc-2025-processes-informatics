#pragma once

#include "kichanova_k_count_letters_in_str/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kichanova_k_count_letters_in_str {

class KichanovaKCountLettersInStrMPI : public BaseTask {
public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KichanovaKCountLettersInStrMPI(const InType &in);

private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

} // namespace kichanova_k_count_letters_in_str
