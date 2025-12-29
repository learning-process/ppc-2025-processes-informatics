#include "morozova_s_broadcast/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "morozova_s_broadcast/common/include/common.hpp"

namespace morozova_s_broadcast {

MorozovaSBroadcastSEQ::MorozovaSBroadcastSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = std::vector<int>();
}

bool MorozovaSBroadcastSEQ::ValidationImpl() {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 1) {
    return false;
  }

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

}  // namespace morozova_s_broadcast
