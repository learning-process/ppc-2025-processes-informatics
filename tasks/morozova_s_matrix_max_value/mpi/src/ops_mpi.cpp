#include "morozova_s_matrix_max_value/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
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
  const auto &mat = GetInput();
  if (mat.empty()) {
    return false;
  }
  for (size_t i = 0; i < mat.size(); ++i) {
    if (mat[i].empty()) {
      return false;
    }
    if (i > 0 && mat[i].size() != mat[0].size()) {
      return false;
    }
  }
  return true;
}

bool MorozovaSMatrixMaxValueMPI::PreProcessingImpl() {
  return true;
}

namespace {

bool CheckMatrixEmpty(const std::vector<std::vector<int>> &mat, int rank) {
  return mat.empty() || (rank == 0 && mat[0].empty());
}

bool ValidateMatrixDimensions(int rows, int cols) {
  return rows <= 0 || cols <= 0;
}

std::vector<int> PrepareMatrixData(int rank, const std::vector<std::vector<int>> &mat, int rows, int cols,
                                   bool &size_mismatch) {
  std::vector<int> data;
  if (rank == 0) {
    data.reserve(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (const auto &row : mat) {
      if (row.size() != static_cast<size_t>(cols)) {
        size_mismatch = true;
        break;
      }
      data.insert(data.end(), row.begin(), row.end());
    }
  }
  return data;
}

void CalculateScatterParameters(int total, int size, std::vector<int> &counts, std::vector<int> &displs) {
  const int base = total / size;
  const int remainder = total % size;
  for (int i = 0, offset = 0; i < size; ++i) {
    const int extra = (i < remainder) ? 1 : 0;
    counts[i] = base + extra;
    displs[i] = offset;
    offset += counts[i];
  }
}

int FindLocalMax(const std::vector<int> &local) {
  int local_max = std::numeric_limits<int>::min();
  for (int value : local) {
    local_max = std::max(local_max, value);
  }
  return local_max;
}

}  // namespace

bool MorozovaSMatrixMaxValueMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto &mat = GetInput();
  if (CheckMatrixEmpty(mat, rank)) {
    GetOutput() = std::numeric_limits<int>::min();
    return true;
  }
  int rows = static_cast<int>(mat.size());
  int cols = static_cast<int>(mat[0].size());
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (ValidateMatrixDimensions(rows, cols)) {
    GetOutput() = std::numeric_limits<int>::min();
    return true;
  }
  bool size_mismatch = false;
  std::vector<int> data = PrepareMatrixData(rank, mat, rows, cols, size_mismatch);
  if (rank == 0 && size_mismatch) {
    rows = 0;
    cols = 0;
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rows == 0 || cols == 0) {
    GetOutput() = std::numeric_limits<int>::min();
    return true;
  }
  const int total = rows * cols;
  std::vector<int> counts(size);
  std::vector<int> displs(size);
  CalculateScatterParameters(total, size, counts, displs);
  const int local_size = counts[rank];
  std::vector<int> local(local_size);
  if (total > 0) {
    void *sendbuf = (rank == 0) ? data.data() : nullptr;
    MPI_Scatterv(sendbuf, counts.data(), displs.data(), MPI_INT, local.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
  }
  int local_max = FindLocalMax(local);
  int global_max = 0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  GetOutput() = global_max;
  return true;
}

bool MorozovaSMatrixMaxValueMPI::PostProcessingImpl() {
  return true;
}

}  // namespace morozova_s_matrix_max_value
