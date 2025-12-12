#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <string>

#include "yurkin_counting_number/common/include/common.hpp"

namespace yurkin_counting_number {

YurkinCountingNumberMPI::YurkinCountingNumberMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinCountingNumberMPI::ValidationImpl() {
  return (GetInput() >= 0) && (GetOutput() == 0);
}

bool YurkinCountingNumberMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YurkinCountingNumberMPI::RunImpl() {
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  const std::string &full = GetInput();
  int n = full.size();

  int chunk = n / world_size;
  int rem = n % world_size;

  int start = world_rank * chunk + std::min(world_rank, rem);
  int size = chunk + (world_rank < rem ? 1 : 0);

  int local = 0;
  for (int i = start; i < start + size; i++) {
    if (std::isalpha(static_cast<unsigned char>(full[i]))) {
      local++;
    }
  }

  int global = 0;
  MPI_Reduce(&local, &global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    GetOutput() = global;
  }

  return true;
}

bool YurkinCountingNumberMPI::PostProcessingImpl() {
  return (GetOutput() >= 0);
}

}  // namespace yurkin_counting_number
