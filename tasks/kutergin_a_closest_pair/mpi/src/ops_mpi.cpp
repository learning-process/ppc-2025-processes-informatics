#include "kutergin_a_closest_pair/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include "kutergin_a_closest_pair/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kutergin_a_closest_pair {

KuterginAClosestPairMPI::KuterginAClosestPairMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = -1;
}

bool KuterginAClosestPairMPI::ValidationImpl() {
  return true;
}

bool KuterginAClosestPairMPI::PreProcessingImpl() {
  return true;
}

bool KuterginAClosestPairMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &v = GetInput();
  int n = v.size();

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n < 2) {
    GetOutput() = -1;
    return true;
  }
  int local_size = n / size;
  int remainder = n % size;

  int start = rank * local_size + std::min(rank, remainder);
  int end = start + local_size + (rank < remainder ? 1 : 0);

  if (rank == size - 1) {
    end = n;
  }

  std::vector<int> local_data(end - start);
  if (rank == 0) {
    std::copy(v.begin() + start, v.begin() + end, local_data.begin());

    for (int i = 1; i < size; ++i) {
      int other_start = i * local_size + std::min(i, remainder);
      int other_end = other_start + local_size + (i < remainder ? 1 : 0);
      if (i == size - 1) {
        other_end = n;
      }

      MPI_Send(v.data() + other_start, other_end - other_start, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_data.data(), local_data.size(), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  int local_min = std::numeric_limits<int>::max();
  int local_idx = -1;

  for (int i = 0; i < static_cast<int>(local_data.size()) - 1; ++i) {
    int diff = std::abs(local_data[i + 1] - local_data[i]);
    if (diff < local_min) {
      local_min = diff;
      local_idx = start + i;
    }
  }

  if (rank < size - 1 && end < n) {
    int boundary_diff = std::abs(v[end] - local_data.back());
    if (boundary_diff < local_min) {
      local_min = boundary_diff;
      local_idx = end - 1;
    }
  }

  struct {
    int val;
    int idx;
  } local{local_min, local_idx}, global;
  MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

  GetOutput() = global.idx;
  return true;
}

bool KuterginAClosestPairMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_a_closest_pair
