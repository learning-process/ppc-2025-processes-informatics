#include "telnov_counting_the_frequency/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "telnov_counting_the_frequency/common/include/common.hpp"
#include "util/include/util.hpp"

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
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::string &s = g_data_string;
  size_t n = s.size();

  size_t chunk = n / size;
  size_t start = rank * chunk;
  size_t end = (rank == size - 1 ? n : start + chunk);

  long long local = 0;
  for (size_t i = start; i < end; i++) {
    if (s[i] == 'X') {
      local++;
    }
  }

  long long total = 0;
  MPI_Allreduce(&local, &total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = total;

  return true;
}

bool TelnovCountingTheFrequencyMPI::PostProcessingImpl() {
  return GetOutput() == GetInput();
}

}  // namespace telnov_counting_the_frequency
