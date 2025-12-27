#include "morozova_s_broadcast/seq/include/ops_seq.hpp"

#include <algorithm>
#include <iterator>
#include <vector>

#include "morozova_s_broadcast/common/include/common.hpp"

namespace morozova_s_broadcast {

MorozovaSBroadcastSEQ::MorozovaSBroadcastSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = std::vector<int>();
}

bool MorozovaSBroadcastSEQ::ValidationImpl() {
  return true;
}

bool MorozovaSBroadcastSEQ::PreProcessingImpl() {
  return true;
}

bool MorozovaSBroadcastSEQ::RunImpl() {
  GetOutput().reserve(GetInput().size());
  std::copy(GetInput().begin(), GetInput().end(), std::back_inserter(GetOutput()));
  return true;
}

bool MorozovaSBroadcastSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace morozova_s_broadcast
