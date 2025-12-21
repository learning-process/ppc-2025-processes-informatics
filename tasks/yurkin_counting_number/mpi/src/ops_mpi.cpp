#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <utility>

#include "yurkin_counting_number/common/include/common.hpp"

namespace yurkin_counting_number {

namespace {

int BroadcastTotalSize(int total_size, int world_rank) {
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return total_size;
}

void BroadcastInputBuffer(InType &local_input, int total_size, int world_rank, const InType &input) {
  local_input.assign(static_cast<size_t>(total_size), '\0');
  if (world_rank == 0) {
    std::ranges::copy(input, local_input.begin());
  }
  MPI_Bcast(total_size > 0 ? local_input.data() : nullptr, total_size, MPI_CHAR, 0, MPI_COMM_WORLD);
}

std::pair<size_t, size_t> ComputeRange(int total_size, int world_size, int world_rank) {
  if (world_size <= 0) {
    return {0u, 0u};
  }
  int chunk = total_size / world_size;
  int rem = total_size % world_size;
  size_t start = static_cast<size_t>(world_rank * chunk + std::min(world_rank, rem));
  size_t size = static_cast<size_t>(chunk + (world_rank < rem ? 1 : 0));
  return {start, size};
}

int CountRange(const InType &data, size_t start, size_t size) {
  int count = 0;
  size_t n = data.size();
  for (size_t idx = start; idx < start + size && idx < n; ++idx) {
    if (std::isalpha(static_cast<unsigned char>(data[idx])) != 0) {
      ++count;
    }
  }
  return count;
}

}  // namespace

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

  int total_size = 0;
  if (world_rank == 0) {
    total_size = static_cast<int>(GetInput().size());
  }

  total_size = BroadcastTotalSize(total_size, world_rank);

  InType local_input;
  BroadcastInputBuffer(local_input, total_size, world_rank, GetInput());

  if (world_rank != 0) {
    GetInput() = local_input;
  }

  auto [start, size] = ComputeRange(total_size, world_size, world_rank);

  int local_count = CountRange(GetInput(), start, size);

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
