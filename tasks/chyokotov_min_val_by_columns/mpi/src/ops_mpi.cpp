#include "chyokotov_min_val_by_columns/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "chyokotov_min_val_by_columns/common/include/common.hpp"

namespace chyokotov_min_val_by_columns {

ChyokotovMinValByColumnsMPI::ChyokotovMinValByColumnsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().clear();
  GetInput().reserve(in.size());
  for (const auto &row : in) {
    GetInput().push_back(row);
  }

  GetOutput().clear();
}

bool ChyokotovMinValByColumnsMPI::ValidationImpl() {
  if (GetInput().empty()) {
    return true;
  }

  size_t length_row = GetInput()[0].size();
  return std::ranges::all_of(GetInput(), [length_row](const auto &row) { return row.size() == length_row; });
}

bool ChyokotovMinValByColumnsMPI::PreProcessingImpl() {
  if (GetInput().empty()) {
    return true;
  }
  GetOutput().resize(GetInput()[0].size(), INT_MAX);
  return true;
}

bool ChyokotovMinValByColumnsMPI::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return true;
  }

  int rank{};
  int size{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t rows = matrix.size();
  size_t cols = matrix[0].size();

  int base = static_cast<int>(cols) / size;
  int rem = static_cast<int>(cols) % size;
  int loc_cols = base + (rank < rem ? 1 : 0);
  int loc_start = (rank * base) + std::min(rank, rem);

  std::vector<int> local_mins(loc_cols, INT_MAX);
  for (int i = 0; i < loc_cols; i++) {
    for (size_t j = 0; j < rows; j++) {
      local_mins[i] = std::min(local_mins[i], matrix[j][loc_start + i]);
    }
  }

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  for (int i = 0; i < size; i++) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = (i * base) + std::min(i, rem);
  }

  MPI_Allgatherv(local_mins.data(), loc_cols, MPI_INT, GetOutput().data(), counts.data(), displs.data(), MPI_INT,
                 MPI_COMM_WORLD);

  return true;
}

bool ChyokotovMinValByColumnsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace chyokotov_min_val_by_columns
