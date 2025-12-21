#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>

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
  int world_size = 0;
  int world_rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  const InType &input = GetInput();
  int total_size = 0;

  if (world_rank == 0) {
    total_size = static_cast<int>(input.size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  InType local_input;
  local_input.assign(total_size, '\0');

  if (world_rank == 0) {
    std::copy(input.begin(), input.end(), local_input.begin());
  }

  if (total_size > 0) {
    MPI_Bcast(local_input.data(), total_size, MPI_CHAR, 0, MPI_COMM_WORLD);
  }

  int chunk = total_size / world_size;
  int rem = total_size % world_size;

  int start = (world_rank * chunk) + std::min(world_rank, rem);
  int size = chunk + (world_rank < rem ? 1 : 0);

  int local_count = 0;
  for (int i = start; i < start + size; ++i) {
    if (std::isalpha(static_cast<unsigned char>(local_input[i])) != 0) {
      local_count++;
    }
  }

  int global_count = 0;
  MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    GetOutput() = global_count;
  }

  return true;
}

bool YurkinCountingNumberMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace yurkin_counting_number
