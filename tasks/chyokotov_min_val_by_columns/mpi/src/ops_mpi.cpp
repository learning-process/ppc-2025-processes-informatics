#include "chyokotov_min_val_by_columns/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "chyokotov_min_val_by_columns/common/include/common.hpp"
#include "util/include/util.hpp"

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

  size_t lengthRow = GetInput()[0].size();
  return std::ranges::all_of(GetInput(), [lengthRow](const auto &row) { return row.size() == lengthRow; });
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

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t rows = matrix.size();
  size_t cols = matrix[0].size();

  size_t base = cols / size;
  int rem = cols % size;
  size_t locCols = base + (rank < rem);
  size_t locStart = rank * base + std::min(rank, rem);

  std::vector<int> local_mins(locCols, INT_MAX);
  for (size_t i = 0; i < locCols; i++) {
    for (size_t j = 0; j < rows; j++) {
      local_mins[i] = std::min(local_mins[i], matrix[j][locStart + i]);
    }
  }

  std::vector<int> counts(size), displs(size);
  for (int i = 0; i < size; i++) {
    counts[i] = base + (i < rem);
    displs[i] = i * base + std::min(i, rem);
  }

  MPI_Allgatherv(local_mins.data(), locCols, MPI_INT, GetOutput().data(), counts.data(), displs.data(), MPI_INT,
                 MPI_COMM_WORLD);

  return true;
}

bool ChyokotovMinValByColumnsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace chyokotov_min_val_by_columns
