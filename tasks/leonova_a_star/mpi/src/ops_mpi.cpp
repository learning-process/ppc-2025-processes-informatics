#include "leonova_a_star/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <vector>

#include "leonova_a_star/common/include/common.hpp"

namespace leonova_a_star {

LeonovaAStarMPI::LeonovaAStarMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool LeonovaAStarMPI::ValidateMatricesOnMaster() {
  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  if (matrix_a.empty() || matrix_b.empty()) {
    return false;
  }

  if (matrix_a[0].empty() || matrix_b[0].empty()) {
    return false;
  }

  size_t rows_a = matrix_a.size();
  size_t cols_a = matrix_a[0].size();

  for (size_t index = 1; index < rows_a; index++) {
    if (matrix_a[index].size() != cols_a) {
      return false;
    }
  }

  size_t rows_b = matrix_b.size();
  size_t cols_b = matrix_b[0].size();

  for (size_t index = 1; index < rows_b; index++) {
    if (matrix_b[index].size() != cols_b) {
      return false;
    }
  }

  return cols_a == rows_b;
}

bool LeonovaAStarMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool is_valid_local = true;

  if (rank == 0) {
    is_valid_local = ValidateMatricesOnMaster();
  }

  int is_valid_int = is_valid_local ? 1 : 0;
  MPI_Bcast(&is_valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return (is_valid_int == 1);
}

bool LeonovaAStarMPI::PreProcessingImpl() {
  return true;
}

namespace {

void DistributeMatrixB(int rank, int cols_a_int, int cols_b_int, const std::vector<std::vector<int>> &matrix_b,
                       std::vector<int> &matrix_b_flat) {
  if (rank == 0) {
    matrix_b_flat.reserve(static_cast<size_t>(cols_a_int) * static_cast<size_t>(cols_b_int));
    for (int index = 0; index < cols_a_int; index++) {
      for (int jndex = 0; jndex < cols_b_int; jndex++) {
        matrix_b_flat.push_back(matrix_b[index][jndex]);
      }
    }
  } else {
    matrix_b_flat.resize(static_cast<size_t>(cols_a_int) * static_cast<size_t>(cols_b_int));
  }

  MPI_Bcast(matrix_b_flat.data(), cols_a_int * cols_b_int, MPI_INT, 0, MPI_COMM_WORLD);
}

void PrepareRowDistribution(int size, int rows_a_int, int cols_a_int, std::vector<int> &rows_per_rank,
                            std::vector<int> &displacements, std::vector<int> &elements_per_rank) {
  int rows_per_process = rows_a_int / size;
  int extra_rows = rows_a_int % size;

  rows_per_rank.resize(size);
  displacements.resize(size, 0);
  elements_per_rank.resize(size);

  for (int index = 0; index < size; index++) {
    rows_per_rank[index] = rows_per_process;
    if (index < extra_rows) {
      rows_per_rank[index] += 1;
    }
  }

  int offset = 0;
  for (int index = 0; index < size; index++) {
    displacements[index] = offset * cols_a_int;
    offset += rows_per_rank[index];
  }

  for (int index = 0; index < size; index++) {
    elements_per_rank[index] = rows_per_rank[index] * cols_a_int;
  }
}

void ScatterMatrixA(int rank, int cols_a_int, const std::vector<std::vector<int>> &matrix_a,
                    const std::vector<int> &displacements, const std::vector<int> &elements_per_rank,
                    std::vector<int> &local_rows_flat) {
  if (rank == 0) {
    std::vector<int> matrix_a_flat;
    matrix_a_flat.reserve(static_cast<size_t>(matrix_a.size()) * static_cast<size_t>(cols_a_int));
    for (const auto &row : matrix_a) {
      matrix_a_flat.insert(matrix_a_flat.end(), row.begin(), row.end());
    }

    MPI_Scatterv(matrix_a_flat.data(), elements_per_rank.data(), displacements.data(), MPI_INT, local_rows_flat.data(),
                 elements_per_rank[rank], MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, elements_per_rank.data(), displacements.data(), MPI_INT, local_rows_flat.data(),
                 elements_per_rank[rank], MPI_INT, 0, MPI_COMM_WORLD);
  }
}

void ComputeLocalMultiplication(int local_rows_count, int cols_a_int, int cols_b_int,
                                const std::vector<std::vector<int>> &local_a,
                                const std::vector<std::vector<int>> &local_b, std::vector<int> &local_result_flat) {
  for (int index = 0; index < local_rows_count; index++) {
    const std::vector<int> &row_a = local_a[index];
    int *result_row = &local_result_flat[static_cast<size_t>(index) * static_cast<size_t>(cols_b_int)];

    for (int kndex = 0; kndex < cols_a_int; kndex++) {
      int aik = row_a[kndex];
      const int *b_row = local_b[kndex].data();

      for (int jndex = 0; jndex < cols_b_int; jndex++) {
        result_row[jndex] += aik * b_row[jndex];
      }
    }
  }
}

void GatherResults(int local_rows_count, int cols_b_int, const std::vector<int> &rows_per_rank,
                   const std::vector<int> &local_result_flat, std::vector<int> &full_result_flat) {
  std::vector<int> result_elements_per_rank(rows_per_rank.size());
  std::vector<int> result_displacements(rows_per_rank.size(), 0);

  for (size_t index = 0; index < rows_per_rank.size(); index++) {
    result_elements_per_rank[index] = rows_per_rank[index] * cols_b_int;
    if (index > 0) {
      result_displacements[index] = result_displacements[index - 1] + result_elements_per_rank[index - 1];
    }
  }

  MPI_Gatherv(local_result_flat.data(), local_rows_count * cols_b_int, MPI_INT, full_result_flat.data(),
              result_elements_per_rank.data(), result_displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);
}

}  // namespace

std::vector<std::vector<int>> LeonovaAStarMPI::MultiplyMatricesMpi(const std::vector<std::vector<int>> &matrix_a,
                                                                   const std::vector<std::vector<int>> &matrix_b) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (matrix_a.empty() || matrix_b.empty() || matrix_a[0].empty() || matrix_b[0].empty()) {
    if (rank == 0) {
      return {};
    }       return std::vector<std::vector<int>>();
   
  }

  size_t rows_a = matrix_a.size();
  size_t cols_a = matrix_a[0].size();
  size_t cols_b = matrix_b[0].size();

  std::array<int, 3> dims{0, 0, 0};
  if (rank == 0) {
    dims[0] = static_cast<int>(rows_a);
    dims[1] = static_cast<int>(cols_a);
    dims[2] = static_cast<int>(cols_b);
  }

  MPI_Bcast(dims.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

  int rows_a_int = dims[0];
  int cols_a_int = dims[1];
  int cols_b_int = dims[2];

  std::vector<int> matrix_b_flat;
  DistributeMatrixB(rank, cols_a_int, cols_b_int, matrix_b, matrix_b_flat);

  std::vector<int> rows_per_rank;
  std::vector<int> displacements;
  std::vector<int> elements_per_rank;
  PrepareRowDistribution(size, rows_a_int, cols_a_int, rows_per_rank, displacements, elements_per_rank);

  int local_rows_count = rows_per_rank[rank];
  std::vector<int> local_rows_flat(elements_per_rank[rank]);

  ScatterMatrixA(rank, cols_a_int, matrix_a, displacements, elements_per_rank, local_rows_flat);

  std::vector<std::vector<int>> local_a(local_rows_count, std::vector<int>(cols_a_int));
  for (int index = 0; index < local_rows_count; index++) {
    for (int jndex = 0; jndex < cols_a_int; jndex++) {
      local_a[index][jndex] = local_rows_flat[(index * cols_a_int) + jndex];
    }
  }

  std::vector<std::vector<int>> local_b(cols_a_int, std::vector<int>(cols_b_int));
  for (int index = 0; index < cols_a_int; index++) {
    for (int jndex = 0; jndex < cols_b_int; jndex++) {
      local_b[index][jndex] = matrix_b_flat[(index * cols_b_int) + jndex];
    }
  }

  std::vector<int> local_result_flat(static_cast<size_t>(local_rows_count) * static_cast<size_t>(cols_b_int), 0);
  ComputeLocalMultiplication(local_rows_count, cols_a_int, cols_b_int, local_a, local_b, local_result_flat);

  std::vector<int> full_result_flat;
  if (rank == 0) {
    full_result_flat.resize(static_cast<size_t>(rows_a_int) * static_cast<size_t>(cols_b_int));
  }

  GatherResults(local_rows_count, cols_b_int, rows_per_rank, local_result_flat, full_result_flat);

  std::vector<std::vector<int>> result;

  if (rank == 0) {
    result.resize(rows_a_int, std::vector<int>(cols_b_int));

    for (int index = 0; index < rows_a_int; index++) {
      for (int jndex = 0; jndex < cols_b_int; jndex++) {
        result[index][jndex] = full_result_flat[(index * cols_b_int) + jndex];
      }
    }
  }

  return result;
}

void LeonovaAStarMPI::BroadcastResult(int rank) {
  int result_rows = 0;
  int result_cols = 0;

  if (rank == 0) {
    result_rows = static_cast<int>(GetOutput().size());
    if (result_rows > 0) {
      result_cols = static_cast<int>(GetOutput()[0].size());
    }
  }

  MPI_Bcast(&result_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&result_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput().resize(result_rows, std::vector<int>(result_cols));

  if (rank == 0) {
    for (int index = 0; index < result_rows; index++) {
      MPI_Bcast(GetOutput()[index].data(), result_cols, MPI_INT, 0, MPI_COMM_WORLD);
    }
  } else {
    for (int index = 0; index < result_rows; index++) {
      GetOutput()[index].resize(result_cols);
      MPI_Bcast(GetOutput()[index].data(), result_cols, MPI_INT, 0, MPI_COMM_WORLD);
    }
  }
}

bool LeonovaAStarMPI::RunImpl() {
  if (!ValidationImpl()) {
    return false;
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  std::vector<std::vector<int>> result = MultiplyMatricesMpi(matrix_a, matrix_b);

  if (rank == 0) {
    GetOutput() = result;
  } else {
    GetOutput().clear();
  }

  BroadcastResult(rank);

  return true;
}

bool LeonovaAStarMPI::PostProcessingImpl() {
  return true;
}

}  // namespace leonova_a_star
