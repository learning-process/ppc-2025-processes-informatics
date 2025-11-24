#include "nikolaev_d_most_dif_vec_neighbors/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "nikolaev_d_most_dif_vec_neighbors/common/include/common.hpp"

namespace nikolaev_d_most_dif_vec_neighbors {

NikolaevDMostDifVecNeighborsMPI::NikolaevDMostDifVecNeighborsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool NikolaevDMostDifVecNeighborsMPI::ValidationImpl() {
  return GetInput().size() >= 2;
}

bool NikolaevDMostDifVecNeighborsMPI::PreProcessingImpl() {
  return true;
}

bool NikolaevDMostDifVecNeighborsMPI::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (input.size() < 2) {
    return false;
  }

  int n = static_cast<int>(input.size());
  int elements_per_process = n / size;
  int remainder = n % size;

  int start = rank * elements_per_process + std::min(rank, remainder);
  int end = start + elements_per_process - 1;
  if (rank < remainder) {
    end++;
  }

  if (rank == size - 1) {
    end = n - 2;
  } else {
    end = std::min(end, n - 2);
  }

  int local_max_diff = -1;
  std::pair<int, int> local_result = {0, 0};

  if (start <= end) {
    for (int i = start; i <= end; i++) {
      int diff = std::abs(input[i + 1] - input[i]);
      if (diff > local_max_diff) {
        local_max_diff = diff;
        local_result = {input[i], input[i + 1]};
      }
    }
  }

  struct LocalMaxStruct {
    int diff = 0;
    int rank = 0;
  };
  LocalMaxStruct local_max;
  LocalMaxStruct global_max;

  local_max.diff = local_max_diff;
  local_max.rank = rank;

  MPI_Allreduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

  std::pair<int, int> global_result;
  if (rank == global_max.rank) {
    global_result = local_result;
  }

  MPI_Bcast(&global_result, 2, MPI_INT, global_max.rank, MPI_COMM_WORLD);

  output = global_result;
  return true;
}

bool NikolaevDMostDifVecNeighborsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace nikolaev_d_most_dif_vec_neighbors
