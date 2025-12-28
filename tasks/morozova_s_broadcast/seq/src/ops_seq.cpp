#include "morozova_s_broadcast/seq/include/ops_seq.hpp"

#include <algorithm>

#include "morozova_s_broadcast/common/include/common.hpp"

namespace morozova_s_broadcast {

MorozovaSBroadcastSEQ::MorozovaSBroadcastSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool MorozovaSBroadcastSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }
  return true;
}

bool MorozovaSBroadcastSEQ::PreProcessingImpl() {
  return true;
}

bool MorozovaSBroadcastSEQ::RunImpl() {
  GetOutput() = GetInput();
  return true;
}

bool MorozovaSBroadcastSEQ::PostProcessingImpl() {
  return true;
}

// Реализация заглушки broadcast для SEQ
void SequentialBroadcast(void *, int, int, int, int) {
  // Ничего делать не нужно
}

}  // namespace morozova_s_broadcast
