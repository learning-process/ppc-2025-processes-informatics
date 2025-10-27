#pragma once

#include "akimov_i_words_string_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace akimov_i_words_string_count {

class AkimovIWordsStringCountMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit AkimovIWordsStringCountMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace akimov_i_words_string_count
