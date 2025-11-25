#include "telnov_counting_the_frequency/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include "telnov_counting_the_frequency/common/include/common.hpp"

namespace telnov_counting_the_frequency {

TelnovCountingTheFrequencyMPI::TelnovCountingTheFrequencyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TelnovCountingTheFrequencyMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool TelnovCountingTheFrequencyMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool TelnovCountingTheFrequencyMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::string &s = GlobalData::g_data_string;
  size_t n = s.size();

  size_t chunk = n / size;
  size_t start = rank * chunk;
  size_t end = (rank == size - 1 ? n : start + chunk);

  int64_t local = 0;
  for (size_t i = start; i < end; i++) {
    if (s[i] == 'X') {
      local++;
    }
  }

  int64_t total = 0;
  MPI_Allreduce(&local, &total, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<int>(total);

  using Clock = std::chrono::high_resolution_clock;
  auto delay_start = Clock::now();
  while (std::chrono::duration<double>(Clock::now() - delay_start).count() < 0.001) {
  }

  return true;
}

bool TelnovCountingTheFrequencyMPI::PostProcessingImpl() {
  return GetOutput() == GetInput();
}

}  // namespace telnov_counting_the_frequency
