#include "sizov_d_string_mismatch_count/mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <mpi.h>

#include <string>
#include <string_view>

#include "sizov_d_string_mismatch_count/common/include/common.hpp"

namespace sizov_d_string_mismatch_count {
SizovDStringMismatchCountMPI::SizovDStringMismatchCountMPI(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput() = 0;
}

bool SizovDStringMismatchCountMPI::ValidationImpl() {
  const auto &input = GetInput();
  const auto &a = std::get<0>(input);
  const auto &b = std::get<1>(input);
  return !a.empty() && a.size() == b.size();
}

bool SizovDStringMismatchCountMPI::PreProcessingImpl() {
  const auto &input = GetInput();
  str_a_ = std::get<0>(input);
  str_b_ = std::get<1>(input);
  return true;
}

bool SizovDStringMismatchCountMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int total_size = static_cast<int>(str_a_.size());
  if (total_size == 0) {
    return true;
  }

  const int base = total_size / size;
  const int remainder = total_size % size;
  const int start = (rank * base) + std::min(rank, remainder);
  const int local_size = base + (rank < remainder ? 1 : 0);

  std::string_view local_a(str_a_.data() + start, local_size);
  std::string_view local_b(str_b_.data() + start, local_size);

  int local_result = 0;
  for (int i = 0; i < local_size; ++i) {
    if (local_a[i] != local_b[i]) {
      ++local_result;
    }
  }

  MPI_Reduce(&local_result, &global_result_, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_result_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_result_;
  return true;
}

bool SizovDStringMismatchCountMPI::PostProcessingImpl() {
  return true;
}
}  // namespace sizov_d_string_mismatch_count
