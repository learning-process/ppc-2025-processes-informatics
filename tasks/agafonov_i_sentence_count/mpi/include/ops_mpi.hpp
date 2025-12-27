#pragma once

#include "agafonov_i_sentence_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace agafonov_i_sentence_count {

class SentenceCountMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SentenceCountMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace agafonov_i_sentence_count
