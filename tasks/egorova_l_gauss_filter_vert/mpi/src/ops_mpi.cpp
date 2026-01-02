#include "egorova_l_gauss_filter_vert/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "egorova_l_gauss_filter_vert/common/include/common.hpp"

namespace egorova_l_gauss_filter_vert {

namespace {

constexpr std::array<float, 9> kKernel = {0.0625F, 0.125F, 0.0625F, 0.125F, 0.25F, 0.125F, 0.0625F, 0.125F, 0.0625F};

void ApplyKernelToPixel(const std::vector<uint8_t> &local_in, int row_idx, int col_idx, int channel_idx, int rows,
                        int local_cols, int channels, double &sum) {
  for (int kernel_row = -1; kernel_row <= 1; ++kernel_row) {
    const int image_row = std::clamp(row_idx + kernel_row, 0, rows - 1);
    for (int kernel_col = -1; kernel_col <= 1; ++kernel_col) {
      const int image_col = std::clamp(col_idx + kernel_col, 0, local_cols - 1);
      const std::size_t pixel_index = ((static_cast<std::size_t>(image_row) * static_cast<std::size_t>(local_cols) +
                                        static_cast<std::size_t>(image_col)) *
                                       static_cast<std::size_t>(channels)) +
                                      static_cast<std::size_t>(channel_idx);
      const auto kernel_index =
          (static_cast<std::size_t>(kernel_row + 1) * 3U) + static_cast<std::size_t>(kernel_col + 1);
      sum += static_cast<double>(local_in[pixel_index]) * static_cast<double>(kKernel.at(kernel_index));
    }
  }
}

void ProcessRow(const std::vector<uint8_t> &local_in, std::vector<uint8_t> &local_out, int row_idx, int rows,
                int local_cols, int channels) {
  for (int col_idx = 0; col_idx < local_cols; ++col_idx) {
    for (int channel_idx = 0; channel_idx < channels; ++channel_idx) {
      double sum = 0.0;
      ApplyKernelToPixel(local_in, row_idx, col_idx, channel_idx, rows, local_cols, channels, sum);

      const std::size_t out_index = ((static_cast<std::size_t>(row_idx) * static_cast<std::size_t>(local_cols) +
                                      static_cast<std::size_t>(col_idx)) *
                                     static_cast<std::size_t>(channels)) +
                                    static_cast<std::size_t>(channel_idx);
      local_out[out_index] = static_cast<uint8_t>(std::clamp(std::round(sum), 0.0, 255.0));
    }
  }
}

void ComputeLocalGauss(const std::vector<uint8_t> &local_in, std::vector<uint8_t> &local_out, int rows, int local_cols,
                       int channels) {
  if (local_cols <= 0) {
    return;
  }

  for (int row_idx = 0; row_idx < rows; ++row_idx) {
    ProcessRow(local_in, local_out, row_idx, rows, local_cols, channels);
  }
}

}  // namespace

EgorovaLGaussFilterVertMPI::EgorovaLGaussFilterVertMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool EgorovaLGaussFilterVertMPI::ValidationImpl() {
  const auto &input = GetInput();
  return input.rows > 0 && input.cols > 0 && input.channels > 0 &&
         input.data.size() == static_cast<std::size_t>(input.rows) * static_cast<std::size_t>(input.cols) *
                                  static_cast<std::size_t>(input.channels);
}

bool EgorovaLGaussFilterVertMPI::PreProcessingImpl() {
  return true;
}

bool EgorovaLGaussFilterVertMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = 0;
  int cols = 0;
  int channels = 0;

  if (rank == 0) {
    rows = GetInput().rows;
    cols = GetInput().cols;
    channels = GetInput().channels;
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int cols_per_proc = cols / size;
  const int remainder = cols % size;

  std::vector<int> send_counts(size);
  std::vector<int> displacements(size);

  if (rank == 0) {
    int offset = 0;
    for (int proc = 0; proc < size; ++proc) {
      const int proc_cols = (proc < remainder) ? (cols_per_proc + 1) : cols_per_proc;
      send_counts[proc] = proc_cols * rows * channels;
      displacements[proc] = offset;
      offset += send_counts[proc];
    }
  }

  MPI_Bcast(send_counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displacements.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  const int local_size = send_counts[rank];
  std::vector<uint8_t> local_in(local_size);
  std::vector<uint8_t> local_out(local_size);

  MPI_Scatterv(rank == 0 ? GetInput().data.data() : nullptr, send_counts.data(), displacements.data(), MPI_BYTE,
               local_in.data(), local_size, MPI_BYTE, 0, MPI_COMM_WORLD);

  const int local_cols = (rows * channels == 0) ? 0 : local_size / (rows * channels);
  ComputeLocalGauss(local_in, local_out, rows, local_cols, channels);

  auto &out = GetOutput();
  out.rows = rows;
  out.cols = cols;
  out.channels = channels;

  const std::size_t total_size =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * static_cast<std::size_t>(channels);
  out.data.resize(total_size);

  MPI_Gatherv(local_out.data(), local_size, MPI_BYTE, out.data.data(), send_counts.data(), displacements.data(),
              MPI_BYTE, 0, MPI_COMM_WORLD);

  MPI_Bcast(out.data.data(), static_cast<int>(total_size), MPI_BYTE, 0, MPI_COMM_WORLD);

  return true;
}

bool EgorovaLGaussFilterVertMPI::PostProcessingImpl() {
  return true;
}

}  // namespace egorova_l_gauss_filter_vert
