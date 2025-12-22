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

  int total_size = 0;
  if (world_rank == 0) {
    total_size = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<char> buffer;
  buffer.assign(static_cast<std::size_t>(total_size), '\0');

  if (world_rank == 0 && total_size > 0) {
    std::copy(GetInput().begin(), GetInput().end(), buffer.begin());
  }

  if (total_size > 0) {
    MPI_Bcast(buffer.data(), total_size, MPI_CHAR, 0, MPI_COMM_WORLD);
  } else {
    MPI_Bcast(nullptr, 0, MPI_CHAR, 0, MPI_COMM_WORLD);
  }

  if (world_rank != 0) {
    GetInput().assign(buffer.begin(), buffer.end());
  }

  const InType &input = GetInput();
  std::size_t n = input.size();

  std::size_t chunk = (world_size > 0) ? (n / static_cast<std::size_t>(world_size)) : 0U;
  std::size_t rem = (world_size > 0) ? (n % static_cast<std::size_t>(world_size)) : 0U;

  auto wrank = static_cast<std::size_t>(world_rank);
  std::size_t start = (wrank * chunk) + std::min(wrank, rem);
  std::size_t end = start + chunk + (wrank < rem ? 1U : 0U);

  start = std::min(start, n);
  end = std::min(end, n);

  int local_count = 0;
  for (std::size_t i = start; i < end; ++i) {
    unsigned char uc = static_cast<unsigned char>(input[i]);

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
