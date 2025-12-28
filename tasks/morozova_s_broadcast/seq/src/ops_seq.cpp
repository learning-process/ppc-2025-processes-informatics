#include "morozova_s_broadcast/seq/include/ops_seq.hpp"

#include "morozova_s_broadcast/common/include/common.hpp"

namespace morozova_s_broadcast {

MorozovaSBroadcastSEQ::MorozovaSBroadcastSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool MorozovaSBroadcastSEQ::ValidationImpl() {
  return !GetInput().empty();
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

void SequentialBroadcast(void *data, int count, int datatype, int root, int comm) {
  (void)data;
  (void)count;
  (void)datatype;
  (void)root;
  (void)comm;
}

}  // namespace morozova_s_broadcast
