#pragma once

#include <vector>

#include "morozova_s_broadcast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace morozova_s_broadcast {

class MorozovaSBroadcastSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit MorozovaSBroadcastSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

// Заглушка для SEQ — объявление
void SequentialBroadcast(void *, int, int, int, int);

}  // namespace morozova_s_broadcast
