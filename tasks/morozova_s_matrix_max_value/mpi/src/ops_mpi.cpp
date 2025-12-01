#include "morozova_s_matrix_max_value/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "morozova_s_matrix_max_value/common/include/common.hpp"

namespace morozova_s_matrix_max_value {

MorozovaSMatrixMaxValueMPI::MorozovaSMatrixMaxValueMPI(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = 0;
}

bool MorozovaSMatrixMaxValueMPI::ValidationImpl() {
  return true;
}

bool MorozovaSMatrixMaxValueMPI::PreProcessingImpl() {
  return true;
}

bool MorozovaSMatrixMaxValueMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto &mat = GetInput();
  int rows = 0;
  int cols = 0;
  bool valid = true;
  rows = mat.size();
  if (rows == 0) {
    valid = false;
  } else {
    cols = mat[0].size();
    for (int i = 1; i < rows; ++i) {
      if (mat[i].size() != cols) {
        valid = false;
        break;
      }
    }
  }
  if (cols <= 0) {
    valid = false;
  }
  int valid_int = valid ? 1 : 0;
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  valid = (valid_int == 1);
  if (!valid) {
    GetOutput() = std::numeric_limits<int>::min();
    return true;
  }
  int total = rows * cols;
  std::vector<int> data;
  if (rank == 0) {
    data.reserve(total);
    for (int i = 0; i < rows; ++i) {
      for (int x : mat[i]) {
        data.push_back(x);
      }
    }
  }
  std::vector<int> counts(size);
  std::vector<int> displs(size);
  int base = total / size;
  int rem = total % size;
  int off = 0;
  for (int i = 0; i < size; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = off;
    off += counts[i];
  }
  int local_size = counts[rank];
  std::vector<int> local(local_size);
  MPI_Scatterv(rank == 0 ? data.data() : nullptr, counts.data(), displs.data(), MPI_INT, local.data(), local_size,
               MPI_INT, 0, MPI_COMM_WORLD);
  int local_max = std::numeric_limits<int>::min();
  for (int x : local) {
    if (x > local_max) {
      local_max = x;
    }
  }
  int global_max = std::numeric_limits<int>::min();
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  GetOutput() = global_max;
  return true;
}

bool MorozovaSMatrixMaxValueMPI::PostProcessingImpl() {
  return true;
}

}  // namespace morozova_s_matrix_max_value
