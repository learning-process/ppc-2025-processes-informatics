#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <numeric>
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

  int total_size = 0;
  if (world_rank == 0) {
    total_size = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<char> buffer;
  if (world_rank == 0 && total_size > 0) {
    buffer.assign(GetInput().begin(), GetInput().end());
  }

  std::vector<int> sendcounts(static_cast<std::size_t>(world_size), 0);
  std::vector<int> displs(static_cast<std::size_t>(world_size), 0);

  if (world_size > 0 && total_size > 0) {
    int base = total_size / world_size;
    int rem = total_size % world_size;
    for (int r = 0; r < world_size; ++r) {
      sendcounts[r] = base + (r < rem ? 1 : 0);
    }
    displs[0] = 0;
    for (int r = 1; r < world_size; ++r) {
      displs[r] = displs[r - 1] + sendcounts[r - 1];
    }
  }

  int local_count_chars = 0;
  if (world_size > 0) {
    local_count_chars = sendcounts[world_rank];
  }
  std::vector<char> local_buf(static_cast<std::size_t>(local_count_chars));

  MPI_Scatterv((buffer.empty() ? nullptr : buffer.data()), sendcounts.data(), displs.data(), MPI_CHAR,
               (local_buf.empty() ? nullptr : local_buf.data()), local_count_chars, MPI_CHAR, 0, MPI_COMM_WORLD);

  int local_count = 0;
  for (char c : local_buf) {
    if (std::isalpha(static_cast<unsigned char>(c)) != 0) {
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
