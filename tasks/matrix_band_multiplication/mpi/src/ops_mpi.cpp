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

  std::array<std::size_t, 4> dims = {rows_a_, cols_a_, rows_b_, cols_b_};
  MPI_Bcast(dims.data(), static_cast<int>(dims.size()), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  rows_a_ = dims[0];
  cols_a_ = dims[1];
  rows_b_ = dims[2];
  cols_b_ = dims[3];

  if (rows_a_ == 0 || cols_a_ == 0 || rows_b_ == 0 || cols_b_ == 0) {
    return false;
  }

  row_counts_ = BuildCounts(static_cast<int>(rows_a_), world_size_);
  row_displs_ = BuildDisplacements(row_counts_);

  std::vector<int> row_send_counts(row_counts_.size(), 0);
  for (std::size_t i = 0; i < row_counts_.size(); ++i) {
    row_send_counts[i] = row_counts_[i] * static_cast<int>(cols_a_);
  }
  std::vector<int> row_send_displs = BuildDisplacements(row_send_counts);

  const double *a_ptr = (rank_ == 0 && !matrix_a.values.empty()) ? matrix_a.values.data() : nullptr;
  const int local_a_elems = row_send_counts[rank_];
  local_a_.assign(static_cast<std::size_t>(local_a_elems), 0.0);

  MPI_Scatterv(a_ptr, row_send_counts.data(), row_send_displs.data(), MPI_DOUBLE, local_a_.data(), local_a_elems,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  col_counts_ = BuildCounts(static_cast<int>(cols_b_), world_size_);
  col_displs_ = BuildDisplacements(col_counts_);
  max_cols_per_proc_ = 0;
  for (int count : col_counts_) {
    if (count > max_cols_per_proc_) {
      max_cols_per_proc_ = count;
    }
  }
  const int buffer_cols = std::max(1, max_cols_per_proc_);
  current_b_.assign(rows_b_ * static_cast<std::size_t>(buffer_cols), 0.0);
  rotation_buffer_.assign(current_b_.size(), 0.0);

  std::vector<int> col_send_counts(col_counts_.size(), 0);
  for (std::size_t i = 0; i < col_counts_.size(); ++i) {
    col_send_counts[i] = col_counts_[i] * static_cast<int>(rows_b_);
  }
  std::vector<int> col_send_displs = BuildDisplacements(col_send_counts);

  std::vector<double> transposed;
  if (rank_ == 0) {
    transposed.resize(rows_b_ * cols_b_);
    for (std::size_t i = 0; i < rows_b_; ++i) {
      for (std::size_t j = 0; j < cols_b_; ++j) {
        transposed[(j * rows_b_) + i] = matrix_b.values[(i * cols_b_) + j];
      }
    }
  }

  const int recv_cols = col_counts_.empty() ? 0 : col_counts_[rank_];
  const int recv_elems = recv_cols * static_cast<int>(rows_b_);
  MPI_Scatterv(transposed.data(), col_send_counts.data(), col_send_displs.data(), MPI_DOUBLE, current_b_.data(),
               recv_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  stripe_owner_ = rank_;
  current_cols_ = recv_cols;

  const auto local_rows = static_cast<std::size_t>(row_counts_[rank_]);
  local_result_.assign(local_rows * cols_b_, 0.0);

  result_counts_ = BuildCounts(static_cast<int>(rows_a_), world_size_);
  result_displs_ = BuildDisplacements(result_counts_);
  for (std::size_t i = 0; i < result_counts_.size(); ++i) {
    result_counts_[i] = result_counts_[i] * static_cast<int>(cols_b_);
  }
  for (std::size_t i = 0; i < result_displs_.size(); ++i) {
    result_displs_[i] = result_displs_[i] * static_cast<int>(cols_b_);
  }

  return true;
}

bool MatrixBandMultiplicationMpi::RunImpl() {
  const int local_rows = row_counts_[rank_];
  if (cols_b_ == 0 || rows_b_ == 0 || local_rows == 0) {
    return true;
  }

  int stripe_owner = stripe_owner_;
  int stripe_cols = current_cols_;

  for (int step = 0; step < world_size_; ++step) {
    const int col_offset = col_displs_.empty() ? 0 : col_displs_[stripe_owner];
    MultiplyStripe(current_b_.data(), stripe_cols, col_offset, local_rows);
    if (world_size_ == 1) {
      break;
    }
    const int send_to = (rank_ - 1 + world_size_) % world_size_;
    const int recv_from = (rank_ + 1) % world_size_;
    const int next_owner = (stripe_owner + 1) % world_size_;
    const int next_cols = col_counts_.empty() ? 0 : col_counts_[next_owner];
    MPI_Sendrecv(current_b_.data(), stripe_cols * static_cast<int>(rows_b_), MPI_DOUBLE, send_to, 0,
                 rotation_buffer_.data(), next_cols * static_cast<int>(rows_b_), MPI_DOUBLE, recv_from, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::swap(current_b_, rotation_buffer_);
    stripe_owner = next_owner;
    stripe_cols = next_cols;
  }

  return true;
}

void MatrixBandMultiplicationMpi::MultiplyStripe(const double *stripe_data, int stripe_cols, int stripe_offset,
                                                 int local_rows) {
  if (stripe_cols == 0 || local_rows == 0) {
    return;
  }
  for (int row = 0; row < local_rows; ++row) {
    const auto a_row_offset = static_cast<std::size_t>(row) * cols_a_;
    const auto result_row_offset = static_cast<std::size_t>(row) * cols_b_;
    const double *a_row = local_a_.data() + a_row_offset;
    double *result_row = local_result_.data() + result_row_offset;
    for (int col = 0; col < stripe_cols; ++col) {
      const double *b_col = stripe_data + (static_cast<std::size_t>(col) * rows_b_);
      double sum = 0.0;
      for (std::size_t k = 0; k < cols_a_; ++k) {
        sum += a_row[k] * b_col[k];
      }
      result_row[stripe_offset + col] = sum;
    }
  }
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

  std::array<std::size_t, 2> dims = {output.rows, output.cols};
  MPI_Bcast(dims.data(), static_cast<int>(dims.size()), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  output.rows = dims[0];
  output.cols = dims[1];

  const std::size_t total_size = output.rows * output.cols;
  output.values.resize(total_size);
  MPI_Bcast(output.values.data(), static_cast<int>(total_size), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return true;
}

}  // namespace matrix_band_multiplication
