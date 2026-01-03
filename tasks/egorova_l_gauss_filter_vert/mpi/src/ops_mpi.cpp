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

void ApplyKernelToPixel(const std::vector<uint8_t> &local_in, int row_index, int col_index, int channel_index, int rows,
                        int local_cols_with_halo, int channels, int halo_left, double &sum) {
  for (int kernel_row = -1; kernel_row <= 1; ++kernel_row) {
    const int image_row = std::clamp(row_index + kernel_row, 0, rows - 1);
    for (int kernel_col = -1; kernel_col <= 1; ++kernel_col) {
      const int local_col_with_halo = col_index + halo_left + kernel_col;
      const int clamped_local_col = std::clamp(local_col_with_halo, 0, local_cols_with_halo - 1);

      const std::size_t pixel_index =
          ((static_cast<std::size_t>(image_row) * static_cast<std::size_t>(local_cols_with_halo) +
            static_cast<std::size_t>(clamped_local_col)) *
           static_cast<std::size_t>(channels)) +
          static_cast<std::size_t>(channel_index);

      const auto kernel_index =
          (static_cast<std::size_t>(kernel_row + 1) * 3U) + static_cast<std::size_t>(kernel_col + 1);

      sum += static_cast<double>(local_in[pixel_index]) * static_cast<double>(kKernel.at(kernel_index));
    }
  }
}

void ProcessRow(const std::vector<uint8_t> &local_in, std::vector<uint8_t> &local_out, int row_index, int rows,
                int local_cols, int local_cols_with_halo, int channels, int halo_left) {
  for (int col_index = 0; col_index < local_cols; ++col_index) {
    for (int channel_index = 0; channel_index < channels; ++channel_index) {
      double sum = 0.0;
      ApplyKernelToPixel(local_in, row_index, col_index, channel_index, rows, local_cols_with_halo, channels, halo_left,
                         sum);

      const std::size_t out_index = ((static_cast<std::size_t>(row_index) * static_cast<std::size_t>(local_cols) +
                                      static_cast<std::size_t>(col_index)) *
                                     static_cast<std::size_t>(channels)) +
                                    static_cast<std::size_t>(channel_index);
      local_out[out_index] = static_cast<uint8_t>(std::clamp(std::round(sum), 0.0, 255.0));
    }
  }
}

void ComputeLocalGaussWithHalo(const std::vector<uint8_t> &local_in_with_halo, std::vector<uint8_t> &local_out,
                               int rows, int local_cols, int local_cols_with_halo, int channels, int halo_left) {
  if (local_cols <= 0) {
    return;
  }

  for (int row_index = 0; row_index < rows; ++row_index) {
    ProcessRow(local_in_with_halo, local_out, row_index, rows, local_cols, local_cols_with_halo, channels, halo_left);
  }
}

void CopyMainDataToHaloBuffer(const std::vector<uint8_t> &local_data, std::vector<uint8_t> &local_data_with_halo,
                              int rows, int local_cols, int local_cols_with_halo, int channels, int halo_size) {
  for (int row = 0; row < rows; ++row) {
    for (int channel = 0; channel < channels; ++channel) {
      for (int local_col = 0; local_col < local_cols; ++local_col) {
        const std::size_t src_idx = ((row * local_cols + local_col) * channels) + channel;
        const std::size_t dst_idx = ((row * local_cols_with_halo + (local_col + halo_size)) * channels) + channel;
        local_data_with_halo[dst_idx] = local_data[src_idx];
      }
    }
  }
}

void FillHaloBoundaries(std::vector<uint8_t> &local_data_with_halo, int rows, int local_cols, int local_cols_with_halo,
                        int channels, int halo_size) {
  for (int row = 0; row < rows; ++row) {
    for (int channel = 0; channel < channels; ++channel) {
      const std::size_t left_idx = ((row * local_cols_with_halo + 0) * channels) + channel;
      const std::size_t first_real_idx = ((row * local_cols_with_halo + halo_size) * channels) + channel;
      const std::size_t right_idx = ((row * local_cols_with_halo + (local_cols_with_halo - 1)) * channels) + channel;
      const std::size_t last_real_idx =
          ((row * local_cols_with_halo + (local_cols + halo_size - 1)) * channels) + channel;

      local_data_with_halo[left_idx] = local_data_with_halo[first_real_idx];
      local_data_with_halo[right_idx] = local_data_with_halo[last_real_idx];
    }
  }
}

std::vector<uint8_t> PrepareLocalDataWithHalo(const std::vector<uint8_t> &local_data, int rows, int local_cols,
                                              int local_cols_with_halo, int channels, int halo_size) {
  std::vector<uint8_t> local_data_with_halo(static_cast<std::size_t>(local_cols_with_halo) *
                                            static_cast<std::size_t>(rows) * static_cast<std::size_t>(channels));

  CopyMainDataToHaloBuffer(local_data, local_data_with_halo, rows, local_cols, local_cols_with_halo, channels,
                           halo_size);

  FillHaloBoundaries(local_data_with_halo, rows, local_cols, local_cols_with_halo, channels, halo_size);

  return local_data_with_halo;
}

void PrepareHaloForSending(const std::vector<uint8_t> &local_data_with_halo, std::vector<uint8_t> &send_left_halo,
                           std::vector<uint8_t> &send_right_halo, int rows, int local_cols, int local_cols_with_halo,
                           int channels, int halo_size) {
  if (local_cols <= 0) {
    return;
  }

  for (int row = 0; row < rows; ++row) {
    for (int channel = 0; channel < channels; ++channel) {
      const std::size_t src_left_idx = ((row * local_cols_with_halo + halo_size) * channels) + channel;
      const std::size_t dst_left_idx = (row * channels) + channel;
      send_left_halo[dst_left_idx] = local_data_with_halo[src_left_idx];

      const std::size_t src_right_idx =
          ((row * local_cols_with_halo + (local_cols + halo_size - 1)) * channels) + channel;
      const std::size_t dst_right_idx = (row * channels) + channel;
      send_right_halo[dst_right_idx] = local_data_with_halo[src_right_idx];
    }
  }
}

void PerformHaloExchange(std::vector<uint8_t> &send_left_halo, std::vector<uint8_t> &send_right_halo,
                         std::vector<uint8_t> &recv_left_halo, std::vector<uint8_t> &recv_right_halo,
                         int halo_column_size, int rank, int size) {
  const bool is_middle_process = rank > 0 && rank < size - 1;
  const bool is_first_process = rank == 0 && size > 1;
  const bool is_last_process = rank == size - 1 && size > 1;

  if (is_middle_process) {
    MPI_Sendrecv(send_left_halo.data(), halo_column_size, MPI_BYTE, rank - 1, 0, recv_right_halo.data(),
                 halo_column_size, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_right_halo.data(), halo_column_size, MPI_BYTE, rank + 1, 0, recv_left_halo.data(),
                 halo_column_size, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else if (is_first_process) {
    MPI_Sendrecv(send_right_halo.data(), halo_column_size, MPI_BYTE, rank + 1, 0, recv_left_halo.data(),
                 halo_column_size, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else if (is_last_process) {
    MPI_Sendrecv(send_left_halo.data(), halo_column_size, MPI_BYTE, rank - 1, 0, recv_right_halo.data(),
                 halo_column_size, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void UpdateHaloZonesWithReceivedData(std::vector<uint8_t> &local_data_with_halo,
                                     const std::vector<uint8_t> &recv_left_halo,
                                     const std::vector<uint8_t> &recv_right_halo, int rows, int local_cols_with_halo,
                                     int channels, int rank, int size) {
  if (rank > 0) {
    for (int row = 0; row < rows; ++row) {
      for (int channel = 0; channel < channels; ++channel) {
        const std::size_t dst_idx = ((row * local_cols_with_halo + 0) * channels) + channel;
        const std::size_t src_idx = (row * channels) + channel;
        local_data_with_halo[dst_idx] = recv_left_halo[src_idx];
      }
    }
  }

  if (rank < size - 1) {
    for (int row = 0; row < rows; ++row) {
      for (int channel = 0; channel < channels; ++channel) {
        const std::size_t dst_idx = ((row * local_cols_with_halo + (local_cols_with_halo - 1)) * channels) + channel;
        const std::size_t src_idx = (row * channels) + channel;
        local_data_with_halo[dst_idx] = recv_right_halo[src_idx];
      }
    }
  }
}

void ExchangeHaloZones(std::vector<uint8_t> &local_data_with_halo, int rows, int local_cols, int local_cols_with_halo,
                       int channels, int halo_size, int rank, int size) {
  if (size <= 1) {
    return;
  }

  const int halo_column_size = rows * channels;

  std::vector<uint8_t> send_left_halo(halo_column_size);
  std::vector<uint8_t> send_right_halo(halo_column_size);
  std::vector<uint8_t> recv_left_halo(halo_column_size);
  std::vector<uint8_t> recv_right_halo(halo_column_size);

  PrepareHaloForSending(local_data_with_halo, send_left_halo, send_right_halo, rows, local_cols, local_cols_with_halo,
                        channels, halo_size);

  PerformHaloExchange(send_left_halo, send_right_halo, recv_left_halo, recv_right_halo, halo_column_size, rank, size);

  UpdateHaloZonesWithReceivedData(local_data_with_halo, recv_left_halo, recv_right_halo, rows, local_cols_with_halo,
                                  channels, rank, size);
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

  std::vector<int> proc_cols_count(size);
  std::vector<int> proc_start_col(size);

  if (rank == 0) {
    int current_col = 0;
    for (int proc = 0; proc < size; ++proc) {
      const bool gets_extra = proc < remainder;
      proc_cols_count[proc] = gets_extra ? cols_per_proc + 1 : cols_per_proc;
      proc_start_col[proc] = current_col;
      current_col += proc_cols_count[proc];
    }
  }

  MPI_Bcast(proc_cols_count.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(proc_start_col.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  const int local_cols = proc_cols_count[rank];

  const int halo_size = 1;
  const int local_cols_with_halo = local_cols + (2 * halo_size);
  const int local_size = local_cols * rows * channels;

  std::vector<int> send_counts(size);
  std::vector<int> displacements(size);

  if (rank == 0) {
    int offset = 0;
    for (int proc = 0; proc < size; ++proc) {
      send_counts[proc] = proc_cols_count[proc] * rows * channels;
      displacements[proc] = offset;
      offset += send_counts[proc];
    }
  }

  std::vector<uint8_t> local_data(local_size);
  std::vector<uint8_t> local_out(local_size);

  const uint8_t *send_buffer = rank == 0 ? GetInput().data.data() : nullptr;
  MPI_Scatterv(send_buffer, send_counts.data(), displacements.data(), MPI_BYTE, local_data.data(), local_size, MPI_BYTE,
               0, MPI_COMM_WORLD);

  auto local_data_with_halo =
      PrepareLocalDataWithHalo(local_data, rows, local_cols, local_cols_with_halo, channels, halo_size);

  ExchangeHaloZones(local_data_with_halo, rows, local_cols, local_cols_with_halo, channels, halo_size, rank, size);

  ComputeLocalGaussWithHalo(local_data_with_halo, local_out, rows, local_cols, local_cols_with_halo, channels,
                            halo_size);

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
