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

  int params[4] = {0, 0, 0, 0};

  if (rank == 0) {
    params[1] = GetInput().width;
    params[2] = GetInput().height;
    params[3] = GetInput().kernel_size;

    if (params[1] > 0 && params[2] > 0 && params[3] > 0) {
      params[0] = 1;
    } else {
      params[0] = 0;
    }
  }

  MPI_Bcast(params, 4, MPI_INT, 0, MPI_COMM_WORLD);

  if (params[0] == 0) {
    return false;
  }

  GetInput().width = params[1];
  GetInput().height = params[2];
  GetInput().kernel_size = params[3];

  return true;
}

bool GaseninLImageSmoothMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() = GetInput();
    std::fill(GetOutput().data.begin(), GetOutput().data.end(), 0);
  } else {
    GetOutput().width = GetInput().width;
    GetOutput().height = GetInput().height;
    GetOutput().kernel_size = GetInput().kernel_size;
  }
  return true;
}

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
      GetOutput().data.resize(static_cast<size_t>(width) * height);
    }
    return true;
  }

  const int overlap_top = radius;
  const int overlap_bottom = radius;

  const int actual_start = std::max(0, start_row - overlap_top);
  const int actual_end = std::min(height, end_row + overlap_bottom);
  const int extended_rows = actual_end - actual_start;
  const int recv_count = extended_rows * width;

  std::vector<uint8_t> local_data(recv_count);

  std::vector<int> sendcounts;
  std::vector<int> displs;

  if (rank == 0) {
    sendcounts.resize(size);
    displs.resize(size);
    for (int i = 0; i < size; ++i) {
      const int r_start = i * base_rows + std::min(i, extra_rows);
      const int r_end = r_start + base_rows + (i < extra_rows ? 1 : 0);
      if (r_end > r_start) {
        const int r_act_start = std::max(0, r_start - overlap_top);
        const int r_act_end = std::min(height, r_end + overlap_bottom);
        sendcounts[i] = (r_act_end - r_act_start) * width;
        displs[i] = r_act_start * width;
      } else {
        sendcounts[i] = 0;
        displs[i] = 0;
      }
    }
  }

  MPI_Scatterv((rank == 0 ? GetInput().data.data() : nullptr), sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
               local_data.data(), recv_count, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  std::vector<uint8_t> local_result(static_cast<size_t>(local_rows) * width);
  const int row_offset_in_buffer = start_row - actual_start;
  const int k_sq = kernel_size * kernel_size;

  for (int i = 0; i < local_rows; ++i) {
    int global_y = start_row + i;
    int buffer_y = row_offset_in_buffer + i;
    bool is_global_border_y = (global_y < radius) || (global_y >= height - radius);

    for (int x = 0; x < width; ++x) {
      int sum = 0;
      bool is_border_x = (x < radius) || (x >= width - radius);

      if (!is_global_border_y && !is_border_x) {
        const uint8_t *row_ptr = local_data.data() + (buffer_y - radius) * width + (x - radius);
        for (int ky = 0; ky < kernel_size; ++ky) {
          for (int kx = 0; kx < kernel_size; ++kx) {
            sum += row_ptr[kx];
          }
          row_ptr += width;
        }
        local_result[i * width + x] = static_cast<uint8_t>(sum / k_sq);
      } else {
        int count = 0;
        for (int ky = -radius; ky <= radius; ++ky) {
          int gy = global_y + ky;
          int valid_gy = Clamp(gy, 0, height - 1);
          int valid_by = valid_gy - actual_start;
          for (int kx = -radius; kx <= radius; ++kx) {
            int gx = Clamp(x + kx, 0, width - 1);
            sum += local_data[valid_by * width + gx];
            count++;
          }
        }
        local_result[i * width + x] = (count > 0) ? static_cast<uint8_t>(sum / count) : 0;
      }
    }
  }

  if (rank == 0) {
    GetOutput().data.resize(static_cast<size_t>(width) * height);
    for (int i = 0; i < size; ++i) {
      const int r_start = i * base_rows + std::min(i, extra_rows);
      const int r_end = r_start + base_rows + (i < extra_rows ? 1 : 0);
      sendcounts[i] = (r_end - r_start) * width;
      displs[i] = r_start * width;
    }
  }

  MPI_Gatherv(local_result.data(), local_rows * width, MPI_UNSIGNED_CHAR,
              (rank == 0 ? GetOutput().data.data() : nullptr), sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0,
              MPI_COMM_WORLD);

  return true;
}

bool GaseninLImageSmoothMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gasenin_l_image_smooth
