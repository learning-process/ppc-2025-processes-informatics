#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "yurkin_counting_number/common/include/common.hpp"

namespace yurkin_counting_number {

YurkinCountingNumberMPI::YurkinCountingNumberMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinCountingNumberMPI::ValidationImpl() {
  return GetOutput() == 0;
}

bool YurkinCountingNumberMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YurkinCountingNumberMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const InType &global_input = GetInput();
  int total_size = 0;

  if (world_rank == 0) {
    total_size = static_cast<int>(global_input.size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (world_size <= 0) {
    GetOutput() = 0;
    return true;
  }

  std::vector<int> sendcounts;
  std::vector<int> displs;

  if (world_rank == 0) {
    sendcounts.assign(static_cast<std::size_t>(world_size), 0);
    displs.assign(static_cast<std::size_t>(world_size), 0);

    const std::size_t n = static_cast<std::size_t>(total_size);
    const std::size_t ws = static_cast<std::size_t>(world_size);
    const std::size_t chunk = (ws > 0U) ? (n / ws) : 0U;
    const std::size_t rem = (ws > 0U) ? (n % ws) : 0U;

    std::size_t offset = 0;
    for (std::size_t i = 0; i < ws; ++i) {
      const std::size_t add = chunk + (i < rem ? 1U : 0U);
      sendcounts[i] = static_cast<int>(add);
      displs[i] = static_cast<int>(offset);
      offset += add;
    }
  }

  int recvcount = 0;

  if (world_rank == 0) {
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &recvcount, 1, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatter(nullptr, 1, MPI_INT, &recvcount, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  if (recvcount < 0) {
    recvcount = 0;
  }

  std::vector<char> local_buffer;
  local_buffer.assign(static_cast<std::size_t>(recvcount), '\0');

  if (world_rank == 0) {
    MPI_Scatterv(global_input.data(), sendcounts.data(), displs.data(), MPI_CHAR, local_buffer.data(), recvcount,
                 MPI_CHAR, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_CHAR, local_buffer.data(), recvcount, MPI_CHAR, 0, MPI_COMM_WORLD);
  }

  int local_count = 0;
  for (int i = 0; i < recvcount; ++i) {
    const auto uc = static_cast<unsigned char>(local_buffer[static_cast<std::size_t>(i)]);
    if ((uc >= 'A' && uc <= 'Z') || (uc >= 'a' && uc <= 'z')) {
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
