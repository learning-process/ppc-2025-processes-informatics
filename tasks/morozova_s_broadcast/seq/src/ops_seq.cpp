#include "morozova_s_broadcast/seq/include/ops_seq.hpp"

#include <mpi.h>

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
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size <= 0) {
    return false;
  }
  if (rank == 0 && GetInput().empty()) {
    return false;
  }
  return true;
}

bool MorozovaSBroadcastSEQ::PreProcessingImpl() {
  return true;
}

bool MorozovaSBroadcastSEQ::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int data_size = 0;
  if (rank == 0) {
    data_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput().resize(data_size);
  if (data_size > 0) {
    if (rank == 0) {
      std::copy(GetInput().begin(), GetInput().end(), GetOutput().begin());
    }
    MPI_Bcast(GetOutput().data(), data_size, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool MorozovaSBroadcastSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace morozova_s_broadcast
