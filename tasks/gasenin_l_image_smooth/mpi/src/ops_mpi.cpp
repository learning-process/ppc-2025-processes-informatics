#include "gasenin_l_image_smooth/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gasenin_l_image_smooth/common/include/common.hpp"

namespace gasenin_l_image_smooth {

GaseninLImageSmoothMPI::GaseninLImageSmoothMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GaseninLImageSmoothMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int width = GetInput().width;
  int height = GetInput().height;
  int kernel_size = GetInput().kernel_size;

  if (rank == 0) {
    if (width <= 0 || height <= 0 || kernel_size <= 0) {
      return false;
    }
  }

  int is_valid = 1;
  if (rank == 0) {
    is_valid = (width > 0 && height > 0 && kernel_size > 0) ? 1 : 0;
  }
  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (!is_valid) {
    return false;
  }

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&kernel_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetInput().width = width;
  GetInput().height = height;
  GetInput().kernel_size = kernel_size;

  return true;
}

bool GaseninLImageSmoothMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() = GetInput();
    GetOutput().data.assign(GetInput().data.size(), 0);
  } else {
    GetOutput().width = GetInput().width;
    GetOutput().height = GetInput().height;
    GetOutput().kernel_size = GetInput().kernel_size;
  }

  return true;
}

namespace {
void ProcessLocalRowsOptimized(int start_row, int end_row, int width, int height, int kernel_size,
                               const std::vector<uint8_t> &input, std::vector<uint8_t> &output) {
  const int radius = kernel_size / 2;

  for (int row_idx = start_row; row_idx < end_row; ++row_idx) {
    for (int col_idx = 0; col_idx < width; ++col_idx) {
      int sum = 0;
      int count = 0;

      if (row_idx >= radius && row_idx < height - radius && col_idx >= radius && col_idx < width - radius) {
        if (kernel_size == 3) {
          const int y_offset_minus = (row_idx - 1) * width;
          const int y_offset = row_idx * width;
          const int y_offset_plus = (row_idx + 1) * width;

          sum = input[y_offset_minus + col_idx - 1] + input[y_offset_minus + col_idx] +
                input[y_offset_minus + col_idx + 1] + input[y_offset + col_idx - 1] + input[y_offset + col_idx] +
                input[y_offset + col_idx + 1] + input[y_offset_plus + col_idx - 1] + input[y_offset_plus + col_idx] +
                input[y_offset_plus + col_idx + 1];
          count = 9;
        } else {
          for (int kernel_y = -radius; kernel_y <= radius; ++kernel_y) {
            const int y_idx = (row_idx + kernel_y) * width;
            for (int kernel_x = -radius; kernel_x <= radius; ++kernel_x) {
              sum += input[y_idx + col_idx + kernel_x];
              ++count;
            }
          }
        }
      } else {
        for (int kernel_y = -radius; kernel_y <= radius; ++kernel_y) {
          for (int kernel_x = -radius; kernel_x <= radius; ++kernel_x) {
            const int ny = Clamp(row_idx + kernel_y, 0, height - 1);
            const int nx = Clamp(col_idx + kernel_x, 0, width - 1);
            sum += input[(ny * width) + nx];
            ++count;
          }
        }
      }

      output[((row_idx - start_row) * width) + col_idx] = static_cast<uint8_t>(sum / count);
    }
  }
}
}  // namespace

bool GaseninLImageSmoothMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int width = GetInput().width;
  const int height = GetInput().height;
  const int kernel_size = GetInput().kernel_size;
  const int radius = kernel_size / 2;

  const int base_rows = height / size;
  const int extra_rows = height % size;

  const int start_row = rank * base_rows + std::min(rank, extra_rows);
  const int end_row = start_row + base_rows + (rank < extra_rows ? 1 : 0);
  const int local_rows = end_row - start_row;

  if (local_rows <= 0) {
    if (rank == 0) {
      GetOutput().data.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    }
    return true;
  }

  std::vector<int> sendcounts(size, 0);
  std::vector<int> displs(size, 0);

  const int overlap_top = (radius > 0) ? radius : 0;
  const int overlap_bottom = (radius > 0) ? radius : 0;

  int total_send_offset = 0;
  for (int i = 0; i < size; ++i) {
    const int i_start = i * base_rows + std::min(i, extra_rows);
    const int i_end = i_start + base_rows + (i < extra_rows ? 1 : 0);
    const int i_rows = i_end - i_start;

    if (i_rows > 0) {
      const int actual_start = std::max(0, i_start - overlap_top);
      const int actual_end = std::min(height, i_end + overlap_bottom);
      const int extended_rows = actual_end - actual_start;

      sendcounts[i] = extended_rows * width;
      displs[i] = actual_start * width;
      total_send_offset += extended_rows * width;
    }
  }

  std::vector<uint8_t> local_data;
  if (local_rows > 0) {
    const int actual_start = std::max(0, start_row - overlap_top);
    const int actual_end = std::min(height, end_row + overlap_bottom);
    const int extended_rows = actual_end - actual_start;
    local_data.resize(static_cast<size_t>(extended_rows) * static_cast<size_t>(width));

    if (rank == 0) {
      for (int i = 0; i < extended_rows; ++i) {
        const int src_row = actual_start + i;
        if (src_row >= 0 && src_row < height) {
          std::copy(GetInput().data.begin() + src_row * width, GetInput().data.begin() + (src_row + 1) * width,
                    local_data.begin() + i * width);
        }
      }
    }

    MPI_Scatterv((rank == 0 ? GetInput().data.data() : nullptr), sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                 local_data.data(), extended_rows * width, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  }

  std::vector<uint8_t> local_result;
  if (local_rows > 0) {
    const int actual_start = std::max(0, start_row - overlap_top);
    const int extended_rows = local_data.size() / width;

    local_result.resize(static_cast<size_t>(local_rows) * static_cast<size_t>(width));

    const int process_start_in_buffer = (start_row - actual_start);
    const int process_end_in_buffer = process_start_in_buffer + local_rows;

    std::vector<uint8_t> temp_full_buffer;
    if (extended_rows > 0) {
      ProcessLocalRowsOptimized(process_start_in_buffer, process_end_in_buffer, width, extended_rows, kernel_size,
                                local_data, local_result);
    }
  }

  std::vector<int> recv_counts(size);
  std::vector<int> recv_displs(size);

  for (int i = 0; i < size; ++i) {
    const int i_start = i * base_rows + std::min(i, extra_rows);
    const int i_end = i_start + base_rows + (i < extra_rows ? 1 : 0);
    const int i_rows = i_end - i_start;

    recv_counts[i] = (i_rows > 0) ? i_rows * width : 0;
    recv_displs[i] = i_start * width;
  }

  if (rank == 0) {
    GetOutput().data.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
  }

  MPI_Gatherv((local_rows > 0 ? local_result.data() : nullptr), local_rows * width, MPI_UNSIGNED_CHAR,
              (rank == 0 ? GetOutput().data.data() : nullptr), recv_counts.data(), recv_displs.data(),
              MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  return true;
}

bool GaseninLImageSmoothMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gasenin_l_image_smooth
