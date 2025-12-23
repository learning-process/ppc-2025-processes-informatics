#include "leonova_a_star/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <new>
#include <utility>
#include <vector>

#include "leonova_a_star/common/include/common.hpp"

namespace leonova_a_star {

namespace {

constexpr size_t kMaxMatrixSize = 10000;
constexpr size_t kMaxVectorSize = 1000000;
constexpr int kMaxProcesses = 1024;

bool CheckMatricesNotEmpty(const std::vector<std::vector<int>> &matrix_a,
                           const std::vector<std::vector<int>> &matrix_b) {
  return !matrix_a.empty() && !matrix_b.empty() && !matrix_a[0].empty() && !matrix_b[0].empty();
}

void FillMatrixBRow(int row_idx, int cols, const std::vector<std::vector<int>> &matrix_b, std::vector<int> &output) {
  if (static_cast<size_t>(row_idx) >= matrix_b.size()) {
    output.insert(output.end(), cols, 0);
    return;
  }

  const auto &row = matrix_b[static_cast<size_t>(row_idx)];
  for (int col_idx = 0; col_idx < cols; ++col_idx) {
    output.push_back(static_cast<size_t>(col_idx) < row.size() ? row[static_cast<size_t>(col_idx)] : 0);
  }
}

void FillMatrixBFlat(int cols_a_int, int cols_b_int, const std::vector<std::vector<int>> &matrix_b,
                     std::vector<int> &matrix_b_flat) {
  for (int index = 0; index < cols_a_int; ++index) {
    FillMatrixBRow(index, cols_b_int, matrix_b, matrix_b_flat);
  }
}

void DistributeMatrixB(int rank, int cols_a_int, int cols_b_int, const std::vector<std::vector<int>> &matrix_b,
                       std::vector<int> &matrix_b_flat) {
  if (rank == 0) {
    matrix_b_flat.reserve(static_cast<size_t>(cols_a_int) * static_cast<size_t>(cols_b_int));
    FillMatrixBFlat(cols_a_int, cols_b_int, matrix_b, matrix_b_flat);
  } else {
    matrix_b_flat.resize(static_cast<size_t>(cols_a_int) * static_cast<size_t>(cols_b_int));
  }

  MPI_Bcast(matrix_b_flat.data(), cols_a_int * cols_b_int, MPI_INT, 0, MPI_COMM_WORLD);
}

bool SafeVectorResize(std::vector<int> &vec, size_t new_size) {
  if (new_size > kMaxVectorSize) {
    return false;
  }

  try {
    vec.resize(new_size);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool SafeVectorResize(std::vector<int> &vec, size_t new_size, int value) {
  if (new_size > kMaxVectorSize) {
    return false;
  }

  try {
    vec.clear();
    vec.insert(vec.end(), new_size, value);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool PrepareRowDistribution(int size, int rows_a_int, int cols_a_int, std::vector<int> &rows_per_rank,
                            std::vector<int> &displacements, std::vector<int> &elements_per_rank) {
  if (rows_a_int <= 0 || cols_a_int <= 0 || size <= 0 || size > kMaxProcesses ||
      std::cmp_greater(static_cast<size_t>(rows_a_int), kMaxMatrixSize)) {
    rows_per_rank.clear();
    displacements.clear();
    elements_per_rank.clear();
    return false;
  }

  int rows_per_process = rows_a_int / size;
  int extra_rows = rows_a_int % size;

  rows_per_rank.clear();
  displacements.clear();
  elements_per_rank.clear();

  rows_per_rank.reserve(static_cast<size_t>(size));
  displacements.reserve(static_cast<size_t>(size));
  elements_per_rank.reserve(static_cast<size_t>(size));

  for (int index = 0; index < size; ++index) {
    int rows = rows_per_process;
    if (index < extra_rows) {
      rows += 1;
    }
    rows_per_rank.push_back(rows);
  }

  int offset = 0;
  for (int index = 0; index < size; ++index) {
    displacements.push_back(offset * cols_a_int);
    offset += rows_per_rank[index];
  }

  for (int index = 0; index < size; ++index) {
    elements_per_rank.push_back(rows_per_rank[index] * cols_a_int);
  }

  return true;
}

void FlattenMatrixA(const std::vector<std::vector<int>> &matrix_a, int cols_a_int, std::vector<int> &matrix_a_flat) {
  const size_t reserve_size = static_cast<size_t>(matrix_a.size()) * static_cast<size_t>(cols_a_int);
  if (reserve_size > kMaxVectorSize) {
    return;
  }

  matrix_a_flat.reserve(reserve_size);

  for (const auto &row : matrix_a) {
    if (std::cmp_greater_equal(row.size(), static_cast<size_t>(cols_a_int))) {
      matrix_a_flat.insert(matrix_a_flat.end(), row.begin(), row.begin() + cols_a_int);
    } else {
      matrix_a_flat.insert(matrix_a_flat.end(), row.begin(), row.end());
      matrix_a_flat.insert(matrix_a_flat.end(), cols_a_int - static_cast<int>(row.size()), 0);
    }
  }
}

void ScatterMatrixA(int rank, int cols_a_int, const std::vector<std::vector<int>> &matrix_a,
                    const std::vector<int> &displacements, const std::vector<int> &elements_per_rank,
                    std::vector<int> &local_rows_flat) {
  if (rank == 0) {
    std::vector<int> matrix_a_flat;
    FlattenMatrixA(matrix_a, cols_a_int, matrix_a_flat);

    if (matrix_a_flat.size() > kMaxVectorSize) {
      for (auto &elem : local_rows_flat) {
        elem = 0;
      }
      return;
    }

    MPI_Scatterv(matrix_a_flat.data(), elements_per_rank.data(), displacements.data(), MPI_INT, local_rows_flat.data(),
                 elements_per_rank[rank], MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, elements_per_rank.data(), displacements.data(), MPI_INT, local_rows_flat.data(),
                 elements_per_rank[rank], MPI_INT, 0, MPI_COMM_WORLD);
  }
}

void MultiplyRowByElement(int value, const std::vector<int> &b_row, int *result_row, int cols) {
  for (int j = 0; j < cols; ++j) {
    if (static_cast<size_t>(j) < b_row.size()) {
      result_row[j] += value * b_row[static_cast<size_t>(j)];
    }
  }
}

void MultiplyRow(const std::vector<int> &row_a, const std::vector<std::vector<int>> &local_b, int *result_row,
                 int cols_a, int cols_b) {
  for (int k = 0; k < cols_a; ++k) {
    if (static_cast<size_t>(k) >= row_a.size() || static_cast<size_t>(k) >= local_b.size()) {
      break;
    }

    int aik = row_a[static_cast<size_t>(k)];
    const auto &b_row = local_b[static_cast<size_t>(k)];
    MultiplyRowByElement(aik, b_row, result_row, cols_b);
  }
}

void ComputeLocalMultiplication(int local_rows_count, int cols_a_int, int cols_b_int,
                                const std::vector<std::vector<int>> &local_a,
                                const std::vector<std::vector<int>> &local_b, std::vector<int> &local_result_flat) {
  if (local_result_flat.empty()) {
    return;
  }

  std::ranges::fill(local_result_flat, 0);

  for (int i = 0; i < local_rows_count && static_cast<size_t>(i) < local_a.size(); ++i) {
    int *result_row = &local_result_flat[static_cast<size_t>(i) * static_cast<size_t>(cols_b_int)];

    if (result_row >= local_result_flat.data() && result_row < local_result_flat.data() + local_result_flat.size()) {
      MultiplyRow(local_a[static_cast<size_t>(i)], local_b, result_row, cols_a_int, cols_b_int);
    }
  }
}

bool PrepareGatherParameters(const std::vector<int> &rows_per_rank, int cols_b_int,
                             std::vector<int> &result_elements_per_rank, std::vector<int> &result_displacements) {
  if (rows_per_rank.empty()) {
    result_elements_per_rank.clear();
    result_displacements.clear();
    return true;
  }

  size_t new_size = rows_per_rank.size();

  if (new_size > kMaxVectorSize) {
    result_elements_per_rank.clear();
    result_displacements.clear();
    return false;
  }

  result_elements_per_rank.clear();
  result_displacements.clear();

  result_elements_per_rank.reserve(new_size);
  result_displacements.reserve(new_size);

  int displacement = 0;
  for (size_t index = 0; index < new_size; ++index) {
    int elements = rows_per_rank[index] * cols_b_int;
    result_elements_per_rank.push_back(elements);
    result_displacements.push_back(displacement);
    displacement += elements;
  }

  return true;
}

void GatherResults(int local_rows_count, int cols_b_int, const std::vector<int> &rows_per_rank,
                   const std::vector<int> &local_result_flat, std::vector<int> &full_result_flat) {
  std::vector<int> result_elements_per_rank;
  std::vector<int> result_displacements;

  if (!PrepareGatherParameters(rows_per_rank, cols_b_int, result_elements_per_rank, result_displacements)) {
    return;
  }

  if (result_elements_per_rank.empty() || result_displacements.empty()) {
    return;
  }

  MPI_Gatherv(local_result_flat.data(), local_rows_count * cols_b_int, MPI_INT, full_result_flat.data(),
              result_elements_per_rank.data(), result_displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);
}

std::vector<std::vector<int>> ConvertFlatToMatrix(const std::vector<int> &flat_data, int rows, int cols) {
  if (rows <= 0 || cols <= 0 || flat_data.empty()) {
    return {};
  }

  size_t expected_size = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (flat_data.size() < expected_size) {
    return {};
  }

  std::vector<std::vector<int>> result;
  try {
    result.resize(static_cast<size_t>(rows), std::vector<int>(cols));
  } catch (const std::bad_alloc &) {
    return {};
  }

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result[static_cast<size_t>(i)][static_cast<size_t>(j)] = flat_data[(i * cols) + j];
    }
  }

  return result;
}

std::pair<bool, std::array<int, 3>> ValidateAndGetDimensions(int rank, int size,
                                                             const std::vector<std::vector<int>> &matrix_a,
                                                             const std::vector<std::vector<int>> &matrix_b) {
  if (size <= 0 || size > kMaxProcesses) {
    return {false, {}};
  }

  if (!CheckMatricesNotEmpty(matrix_a, matrix_b)) {
    return {false, {}};
  }

  size_t rows_a = matrix_a.size();
  size_t cols_a = matrix_a[0].size();
  size_t cols_b = matrix_b[0].size();

  if (rows_a > kMaxMatrixSize || cols_a > kMaxMatrixSize || cols_b > kMaxMatrixSize) {
    return {false, {}};
  }

  std::array<int, 3> dims{0, 0, 0};
  if (rank == 0) {
    dims[0] = static_cast<int>(rows_a);
    dims[1] = static_cast<int>(cols_a);
    dims[2] = static_cast<int>(cols_b);
  }

  MPI_Bcast(dims.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
  return {true, dims};
}

bool ValidateDimensions(int rows_a_int, int cols_a_int, int cols_b_int) {
  if (rows_a_int <= 0 || cols_a_int <= 0 || cols_b_int <= 0) {
    return false;
  }

  if (std::cmp_greater(static_cast<size_t>(rows_a_int), kMaxMatrixSize) ||
      std::cmp_greater(static_cast<size_t>(cols_a_int), kMaxMatrixSize) ||
      std::cmp_greater(static_cast<size_t>(cols_b_int), kMaxMatrixSize)) {
    return false;
  }

  return true;
}

bool PrepareLocalData(int rank, int cols_a_int, int cols_b_int, const std::vector<std::vector<int>> &matrix_b,
                      std::vector<int> &matrix_b_flat, const std::vector<int> &elements_per_rank,
                      std::vector<int> &local_rows_flat) {
  DistributeMatrixB(rank, cols_a_int, cols_b_int, matrix_b, matrix_b_flat);

  try {
    local_rows_flat.reserve(static_cast<size_t>(elements_per_rank[static_cast<size_t>(rank)]));
  } catch (const std::bad_alloc &) {
    return false;
  }

  return SafeVectorResize(local_rows_flat, static_cast<size_t>(elements_per_rank[static_cast<size_t>(rank)]));
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> CreateLocalMatrices(
    int local_rows_count, int cols_a_int, int cols_b_int, const std::vector<int> &local_rows_flat,
    const std::vector<int> &matrix_b_flat) {
  std::vector<std::vector<int>> local_a;
  try {
    local_a.resize(static_cast<size_t>(local_rows_count));
    for (auto &row : local_a) {
      row.resize(static_cast<size_t>(cols_a_int), 0);
    }
  } catch (const std::bad_alloc &) {
    return {{}, {}};
  }

  for (int i = 0; i < local_rows_count; ++i) {
    for (int j = 0; j < cols_a_int; ++j) {
      local_a[static_cast<size_t>(i)][static_cast<size_t>(j)] = local_rows_flat[(i * cols_a_int) + j];
    }
  }

  std::vector<std::vector<int>> local_b;
  try {
    local_b.resize(static_cast<size_t>(cols_a_int));
    for (auto &row : local_b) {
      row.resize(static_cast<size_t>(cols_b_int), 0);
    }
  } catch (const std::bad_alloc &) {
    return {{}, {}};
  }

  for (int i = 0; i < cols_a_int; ++i) {
    for (int j = 0; j < cols_b_int; ++j) {
      local_b[static_cast<size_t>(i)][static_cast<size_t>(j)] = matrix_b_flat[(i * cols_b_int) + j];
    }
  }

  return {std::move(local_a), std::move(local_b)};
}

std::vector<int> ComputeLocalResult(int local_rows_count, int cols_a_int, int cols_b_int,
                                    const std::vector<std::vector<int>> &local_a,
                                    const std::vector<std::vector<int>> &local_b) {
  std::vector<int> local_result_flat;
  size_t local_result_size = static_cast<size_t>(local_rows_count) * static_cast<size_t>(cols_b_int);
  if (!SafeVectorResize(local_result_flat, local_result_size, 0)) {
    return {};
  }

  ComputeLocalMultiplication(local_rows_count, cols_a_int, cols_b_int, local_a, local_b, local_result_flat);
  return local_result_flat;
}

}  // namespace

LeonovaAStarMPI::LeonovaAStarMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool LeonovaAStarMPI::ValidateMatricesOnMaster() {
  const auto &matrix_a = std::get<0>(GetInput());
  const auto &matrix_b = std::get<1>(GetInput());

  if (!CheckMatricesNotEmpty(matrix_a, matrix_b)) {
    return false;
  }

  if (matrix_a.size() > kMaxMatrixSize || matrix_b.size() > kMaxMatrixSize) {
    return false;
  }

  size_t rows_a = matrix_a.size();
  size_t cols_a = matrix_a[0].size();

  for (size_t index = 1; index < rows_a; ++index) {
    if (index >= matrix_a.size() || matrix_a[index].size() != cols_a) {
      return false;
    }
  }

  size_t rows_b = matrix_b.size();
  size_t cols_b = matrix_b[0].size();

  for (size_t index = 1; index < rows_b; ++index) {
    if (index >= matrix_b.size() || matrix_b[index].size() != cols_b) {
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

std::vector<std::vector<int>> LeonovaAStarMPI::MultiplyMatricesMpi(const std::vector<std::vector<int>> &matrix_a,
                                                                   const std::vector<std::vector<int>> &matrix_b) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto [valid, dims] = ValidateAndGetDimensions(rank, size, matrix_a, matrix_b);
  if (!valid) {
    return {};
  }

  int rows_a_int = dims[0];
  int cols_a_int = dims[1];
  int cols_b_int = dims[2];

  if (!::leonova_a_star::ValidateDimensions(rows_a_int, cols_a_int, cols_b_int)) {
    return {};
  }

  std::vector<int> rows_per_rank;
  std::vector<int> displacements;
  std::vector<int> elements_per_rank;
  if (!PrepareRowDistribution(size, rows_a_int, cols_a_int, rows_per_rank, displacements, elements_per_rank)) {
    return {};
  }

  if (rows_per_rank.empty()) {
    return {};
  }

  int local_rows_count = rows_per_rank[static_cast<size_t>(rank)];
  if (local_rows_count < 0) {
    return {};
  }

  std::vector<int> matrix_b_flat;
  std::vector<int> local_rows_flat;
  if (!PrepareLocalData(rank, cols_a_int, cols_b_int, matrix_b, matrix_b_flat, elements_per_rank, local_rows_flat)) {
    return {};
  }

  ScatterMatrixA(rank, cols_a_int, matrix_a, displacements, elements_per_rank, local_rows_flat);
  auto [local_a, local_b] =
      CreateLocalMatrices(local_rows_count, cols_a_int, cols_b_int, local_rows_flat, matrix_b_flat);
  if (local_a.empty() || local_b.empty()) {
    return {};
  }

  std::vector<int> local_result_flat = ComputeLocalResult(local_rows_count, cols_a_int, cols_b_int, local_a, local_b);
  if (local_result_flat.empty()) {
    return {};
  }

  std::vector<int> full_result_flat;
  if (rank == 0) {
    size_t full_result_size = static_cast<size_t>(rows_a_int) * static_cast<size_t>(cols_b_int);
    if (!SafeVectorResize(full_result_flat, full_result_size)) {
      return {};
    }
  }

  GatherResults(local_rows_count, cols_b_int, rows_per_rank, local_result_flat, full_result_flat);

  if (rank == 0) {
    return ConvertFlatToMatrix(full_result_flat, rows_a_int, cols_b_int);
  }

  return {};
}

bool LeonovaAStarMPI::ResizeOutputMatrix(int rows, int cols) {
  if (rows < 0 || cols < 0) {
    GetOutput().clear();
    return false;
  }

  if (std::cmp_greater(static_cast<size_t>(rows), kMaxMatrixSize) ||
      std::cmp_greater(static_cast<size_t>(cols), kMaxMatrixSize)) {
    GetOutput().clear();
    return false;
  }

  try {
    GetOutput().resize(static_cast<size_t>(rows));
    for (auto &row : GetOutput()) {
      row.resize(static_cast<size_t>(cols), 0);
    }
    return true;
  } catch (const std::bad_alloc &) {
    GetOutput().clear();
    return false;
  }
}

std::pair<int, int> LeonovaAStarMPI::GetResultDimensions(int rank) {
  int rows = 0;
  int cols = 0;

  if (rank == 0) {
    rows = static_cast<int>(GetOutput().size());
    if (rows > 0 && !GetOutput().empty()) {
      cols = static_cast<int>(GetOutput()[0].size());
    }
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return {rows, cols};
}

bool LeonovaAStarMPI::ValidateDimensions(int rows, int cols) {
  return rows >= 0 && cols >= 0;
}

void LeonovaAStarMPI::BroadcastFromMaster(int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    if (std::cmp_less(i, GetOutput().size()) && !GetOutput()[static_cast<size_t>(i)].empty()) {
      MPI_Bcast(GetOutput()[static_cast<size_t>(i)].data(), cols, MPI_INT, 0, MPI_COMM_WORLD);
    }
  }
}

void LeonovaAStarMPI::ReceiveFromMaster(int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    if (std::cmp_less(i, GetOutput().size())) {
      MPI_Bcast(GetOutput()[static_cast<size_t>(i)].data(), cols, MPI_INT, 0, MPI_COMM_WORLD);
    }
  }
}

void LeonovaAStarMPI::BroadcastResult(int rank) {
  auto [result_rows, result_cols] = GetResultDimensions(rank);

  if (!ValidateDimensions(result_rows, result_cols)) {
    GetOutput().clear();
    return;
  }

  if (!ResizeOutputMatrix(result_rows, result_cols)) {
    return;
  }

  if (rank == 0) {
    BroadcastFromMaster(result_rows, result_cols);
  } else {
    ReceiveFromMaster(result_rows, result_cols);
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
    GetOutput() = std::move(result);
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
