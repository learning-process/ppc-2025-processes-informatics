#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

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
  int rank = 0;
  int size = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> sendcounts(size, 0);
  std::vector<int> displs(size, 0);

  const int base = (size > 0) ? total_size / size : 0;
  const int rem = (size > 0) ? total_size % size : 0;

  for (int i = 0; i < size; ++i) {
    sendcounts[i] = base + (i < rem ? 1 : 0);
  }

  std::partial_sum(sendcounts.begin(), sendcounts.end() - 1, displs.begin() + 1);

  std::vector<char> local_buf(static_cast<std::size_t>(sendcounts[rank]));

  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, sendcounts.data(), displs.data(), MPI_CHAR, local_buf.data(),
               sendcounts[rank], MPI_CHAR, 0, MPI_COMM_WORLD);

  int local_count = 0;
  for (unsigned char c : local_buf) {
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
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
