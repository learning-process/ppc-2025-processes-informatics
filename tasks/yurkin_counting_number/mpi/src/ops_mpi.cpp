#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>

#include "yurkin_counting_number/common/include/common.hpp"

namespace yurkin_counting_number {

YurkinCountingNumberMPI::YurkinCountingNumberMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinCountingNumberMPI::ValidationImpl() {
  return !GetInput().empty() && GetOutput() == 0;
}

bool YurkinCountingNumberMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YurkinCountingNumberMPI::RunImpl() {
  (void)GetOutput();

  int world_rank = 0;
  int world_size = 0;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) != MPI_SUCCESS) {
    return false;
  }
  if (MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS) {
    return false;
  }

  const InType &input = GetInput();
  std::uint64_t total_size = static_cast<std::uint64_t>(input.size());

  std::uint64_t n = (world_rank == 0) ? total_size : 0;
  if (MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
    return false;
  }

  if (n > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  int n_int = static_cast<int>(n);

  if (world_rank != 0) {
    GlobalData::g_data_string.clear();
    GlobalData::g_data_string.resize(static_cast<size_t>(n), '\0');
  } else {
    GlobalData::g_data_string.resize(static_cast<size_t>(n));
    for (std::size_t i = 0; i < input.size() && i < GlobalData::g_data_string.size(); ++i) {
      GlobalData::g_data_string[i] = input[i];
    }
  }

  char *buf = GlobalData::g_data_string.empty() ? nullptr : &GlobalData::g_data_string[0];
  if (n_int > 0) {
    if (MPI_Bcast(buf, n_int, MPI_CHAR, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
      return false;
    }
  }

  const std::string &s = GlobalData::g_data_string;
  int total_len = static_cast<int>(s.size());

  int chunk = total_len / world_size;
  int rem = total_len % world_size;

  int start = world_rank * chunk + std::min(world_rank, rem);
  int size = chunk + (world_rank < rem ? 1 : 0);

  long long local_count = 0;
  for (int i = start; i < start + size; ++i) {
    if (i >= 0 && i < total_len) {
      if (std::isalpha(static_cast<unsigned char>(s[i])) != 0) {
        ++local_count;
      }
    }
  }

  long long global_count = 0;
  if (MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
    return false;
  }

  if (world_rank == 0) {
    if (global_count < static_cast<long long>(std::numeric_limits<int>::min()) ||
        global_count > static_cast<long long>(std::numeric_limits<int>::max())) {
      return false;
    }
    GetOutput() = static_cast<int>(global_count);
  }

  return true;
}

bool YurkinCountingNumberMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace yurkin_counting_number
