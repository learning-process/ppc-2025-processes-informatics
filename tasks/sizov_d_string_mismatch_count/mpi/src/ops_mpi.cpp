#include "sizov_d_string_mismatch_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <string>

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
  global_result_ = 0;
  return true;
}

bool SizovDStringMismatchCountMPI::RunImpl() {
  int mpi_inited = 0;
  int mpi_finalized = 0;
  MPI_Initialized(&mpi_inited);
  MPI_Finalized(&mpi_finalized);
  if (mpi_inited == 0 || mpi_finalized != 0) {
    return false;
  }

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string local_a;
  std::string local_b;

  if (rank == 0) {
    const int n = static_cast<int>(str_a_.size());
    const int base = n / size;
    const int rem = n % size;

    const int my_count = base + (rem > 0 ? 1 : 0);
    local_a = str_a_.substr(0, my_count);
    local_b = str_b_.substr(0, my_count);

    int offset = my_count;
    for (int p = 1; p < size; ++p) {
      const int part = base + (p < rem ? 1 : 0);

      MPI_Send(&part, 1, MPI_INT, p, 0, MPI_COMM_WORLD);

      if (part > 0) {
        MPI_Send(str_a_.data() + offset, part, MPI_CHAR, p, 1, MPI_COMM_WORLD);
        MPI_Send(str_b_.data() + offset, part, MPI_CHAR, p, 2, MPI_COMM_WORLD);
        offset += part;
      }
    }
  } else {
    int part = 0;
    MPI_Recv(&part, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (part > 0) {
      local_a.resize(static_cast<std::size_t>(part));
      local_b.resize(static_cast<std::size_t>(part));

      MPI_Recv(local_a.data(), part, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(local_b.data(), part, MPI_CHAR, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  int local_result = 0;
  for (std::size_t i = 0; i < local_a.size(); ++i) {
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
