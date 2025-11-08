#include "kurpiakov_a_elem_vec_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "kurpiakov_a_elem_vec_sum/common/include/common.hpp"

namespace kurpiakov_a_elem_vec_sum {
KurpiakovAElemVecSumMPI::KurpiakovAElemVecSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool KurpiakovAElemVecSumMPI::ValidationImpl() {
  bool if_dividable = std::get<1>(GetInput()).size() % std::get<0>(GetInput()) == 0;
  bool res = (GetOutput() == 0.0) && if_dividable;
  return res;
}

bool KurpiakovAElemVecSumMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool KurpiakovAElemVecSumMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(std::get<0>(GetInput()));
  }

  if (total_size == 0) {
    GetOutput() = 0.0;
    return true;
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  int base = total_size / size;
  int rem = total_size % size;
  for (int i = 0; i < size; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = (i == 0) ? 0 : (displs[i - 1] + counts[i - 1]);
  }

  std::vector<double> local_vec;
  int my_count = counts[rank];
  local_vec.clear();
  if (my_count > 0) {
    local_vec.resize(my_count);
  }

  MPI_Scatterv(rank == 0 ? &std::get<1>(GetInput()) : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
               my_count > 0 ? local_vec.data() : nullptr, my_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  OutType local_sum = 0.0;
  if (!local_vec.empty()) {
    local_sum = std::accumulate(local_vec.begin(), local_vec.end(), 0.0);
  }

  OutType global_sum = 0.0;

  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_sum;
  } else {
    GetOutput() = 0.0;
  }

  return true;
}

bool KurpiakovAElemVecSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kurpiakov_a_elem_vec_sum
