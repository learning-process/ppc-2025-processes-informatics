#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "yurkin_counting_number/common/include/common.hpp"

namespace yurkin_counting_number {

namespace {

std::pair<std::vector<int>, std::vector<int>> ComputeSendCountsAndDispls(int total_size, int world_size) {
  std::vector<int> sendcounts(static_cast<std::size_t>(world_size), 0);
  std::vector<int> displs(static_cast<std::size_t>(world_size), 0);

  if (world_size <= 0 || total_size <= 0) {
    return {sendcounts, displs};
  }

  const std::size_t n = static_cast<std::size_t>(total_size);
  const std::size_t ws = static_cast<std::size_t>(world_size);
  const std::size_t chunk = n / ws;
  const std::size_t rem = n % ws;

  std::size_t offset = 0;
  for (std::size_t i = 0; i < ws; ++i) {
    const std::size_t add = chunk + (i < rem ? 1 : 0);
    sendcounts[i] = static_cast<int>(add);
    displs[i] = static_cast<int>(offset);
    offset += add;
  }

  return {sendcounts, displs};
}

}  // namespace

YurkinCountingNumberMPI::YurkinCountingNumberMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinCountingNumberMPI::ValidationImpl() {
  return true;
}

bool YurkinCountingNumberMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YurkinCountingNumberMPI::RunImpl() {
  int world_rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const InType &global_input = GetInput();
  int total_size = 0;

  if (world_rank == 0) {
    total_size = static_cast<int>(global_input.size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (world_size <= 0 || total_size <= 0) {
    GetOutput() = 0;
    return true;
  }

  std::vector<int> sendcounts, displs;
  std::tie(sendcounts, displs) = ComputeSendCountsAndDispls(total_size, world_size);

  int recvcount = (world_rank < static_cast<int>(sendcounts.size())) ? sendcounts[world_rank] : 0;
  recvcount = std::max(recvcount, 0);

  std::vector<char> local_buffer(recvcount);

  const void *sendbuf = (world_rank == 0) ? static_cast<const void *>(global_input.data()) : nullptr;
  const int *sendcounts_ptr = (world_rank == 0) ? sendcounts.data() : nullptr;
  const int *displs_ptr = (world_rank == 0) ? displs.data() : nullptr;
  void *recvbuf = (recvcount > 0) ? static_cast<void *>(local_buffer.data()) : nullptr;

  MPI_Scatterv(sendbuf, sendcounts_ptr, displs_ptr, MPI_CHAR, recvbuf, recvcount, MPI_CHAR, 0, MPI_COMM_WORLD);

  int local_count = 0;
  for (int i = 0; i < recvcount; ++i) {
    const auto uc = static_cast<unsigned char>(local_buffer[i]);
    if (std::isalpha(uc)) {
      ++local_count;
    }
  }

  int global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_count;
  return true;
}

bool YurkinCountingNumberMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace yurkin_counting_number
