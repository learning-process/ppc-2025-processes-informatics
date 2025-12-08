#include "matrix_band_multiplication/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "matrix_band_multiplication/common/include/common.hpp"

namespace matrix_band_multiplication {

namespace {
std::vector<int> BuildCounts(int total, int parts) {
  std::vector<int> counts(parts, 0);
  if (parts <= 0) {
    return counts;
  }
  const int base = total / parts;
  int remainder = total % parts;
  for (int i = 0; i < parts; ++i) {
    counts[i] = base + (remainder > 0 ? 1 : 0);
    if (remainder > 0) {
      --remainder;
    }
  }
  return counts;
}

std::vector<int> BuildDisplacements(const std::vector<int> &counts) {
  std::vector<int> displs(counts.size(), 0);
  for (std::size_t i = 1; i < counts.size(); ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }
  return displs;
}

bool MatrixIsValid(const Matrix &matrix) {
  return matrix.rows > 0 && matrix.cols > 0 && matrix.values.size() == matrix.rows * matrix.cols;
}

}  // namespace

MatrixBandMultiplicationMpi::MatrixBandMultiplicationMpi(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = Matrix{};
}

bool MatrixBandMultiplicationMpi::ValidationImpl() {
  const auto &matrix_a = GetInput().a;
  const auto &matrix_b = GetInput().b;
  if (!MatrixIsValid(matrix_a) || !MatrixIsValid(matrix_b)) {
    return false;
  }
  return matrix_a.cols == matrix_b.rows;
}

bool MatrixBandMultiplicationMpi::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  const auto &matrix_a = GetInput().a;
  const auto &matrix_b = GetInput().b;
  if (rank_ == 0) {
    rows_a_ = matrix_a.rows;
    cols_a_ = matrix_a.cols;
    rows_b_ = matrix_b.rows;
    cols_b_ = matrix_b.cols;
  }

  std::array<std::size_t, 4> dims = {rows_a_, cols_a_, rows_b_, cols_b_};                            // GCOVR_EXCL_LINE
  MPI_Bcast(dims.data(), static_cast<int>(dims.size()), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);  // GCOVR_EXCL_LINE
  rows_a_ = dims[0];
  cols_a_ = dims[1];
  rows_b_ = dims[2];
  cols_b_ = dims[3];

  if (rows_a_ == 0 || cols_a_ == 0 || rows_b_ == 0 || cols_b_ == 0) {
    return false;
  }

  row_counts_ = BuildCounts(static_cast<int>(rows_a_), world_size_);
  row_displs_ = BuildDisplacements(row_counts_);

  std::vector<int> send_counts(row_counts_.size());
  std::ranges::transform(row_counts_, send_counts.begin(),
                         [this](int rows) { return rows * static_cast<int>(cols_a_); });
  std::vector<int> send_displs = BuildDisplacements(send_counts);

  const double *a_ptr = rank_ == 0 ? matrix_a.values.data() : nullptr;
  const int local_elems = send_counts[rank_];
  local_a_.assign(static_cast<std::size_t>(local_elems), 0.0);

  MPI_Scatterv(a_ptr, send_counts.data(), send_displs.data(), MPI_DOUBLE, local_a_.data(), local_elems, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  full_b_.resize(rows_b_ * cols_b_);
  if (rank_ == 0) {
    std::ranges::copy(matrix_b.values, full_b_.begin());
  }
  MPI_Bcast(full_b_.data(), static_cast<int>(rows_b_ * cols_b_), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  const auto local_elems_result = static_cast<std::size_t>(row_counts_[rank_]) * cols_b_;
  local_result_.assign(local_elems_result, 0.0);

  result_counts_ = BuildCounts(static_cast<int>(rows_a_), world_size_);
  result_displs_ = BuildDisplacements(result_counts_);
  std::ranges::transform(result_counts_, result_counts_.begin(),
                         [this](int rows) { return rows * static_cast<int>(cols_b_); });
  std::ranges::transform(result_displs_, result_displs_.begin(),
                         [this](int rows_prefix) { return rows_prefix * static_cast<int>(cols_b_); });

  return true;
}

bool MatrixBandMultiplicationMpi::RunImpl() {
  const int local_rows = row_counts_[rank_];
  for (int i = 0; i < local_rows; ++i) {
    for (std::size_t j = 0; j < cols_b_; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k < cols_a_; ++k) {
        const std::size_t a_idx = (static_cast<std::size_t>(i) * cols_a_) + k;
        const std::size_t b_idx = (k * cols_b_) + j;
        sum += local_a_[a_idx] * full_b_[b_idx];
      }
      local_result_[(static_cast<std::size_t>(i) * cols_b_) + j] = sum;
    }
  }
  return true;
}

bool MatrixBandMultiplicationMpi::PostProcessingImpl() {
  std::vector<double> gathered;
  if (rank_ == 0) {
    gathered.resize(rows_a_ * cols_b_);
  }

  const int local_result_elements = row_counts_[rank_] * static_cast<int>(cols_b_);
  MPI_Gatherv(local_result_.data(), local_result_elements, MPI_DOUBLE, gathered.data(), result_counts_.data(),
              result_displs_.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  auto &output = GetOutput();
  if (rank_ == 0) {
    output.rows = rows_a_;
    output.cols = cols_b_;
    output.values = std::move(gathered);
  }

  std::array<std::size_t, 2> dims = {output.rows, output.cols};                                      // GCOVR_EXCL_LINE
  MPI_Bcast(dims.data(), static_cast<int>(dims.size()), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);  // GCOVR_EXCL_LINE
  output.rows = dims[0];
  output.cols = dims[1];

  const std::size_t total_size = output.rows * output.cols;
  output.values.resize(total_size);
  MPI_Bcast(output.values.data(), static_cast<int>(total_size), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return true;
}

}  // namespace matrix_band_multiplication
