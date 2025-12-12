#include "smyshlaev_a_gauss_filt/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "smyshlaev_a_gauss_filt/common/include/common.hpp"
#include "util/include/util.hpp"

namespace smyshlaev_a_gauss_filt {

namespace {
  const std::vector<int> kernel = {
      1, 2, 1,
      2, 4, 2,
      1, 2, 1
  };
  const int kernel_sum = 16;

  struct BlockInfo {
    int start_row;
    int start_col;
    int block_height;
    int block_width;
    int padded_height;  // высота с перекрытием
    int padded_width;   // ширина с перекрытием
    int offset;         // смещение в скаттер-буфере
    int count;          // количество элементов для этого процесса
  };

  void FindOptimalGrid(int size, int& grid_rows, int& grid_cols) {
    int best_diff = size;
    grid_rows = 1;
    grid_cols = size;
    
    for (int rows = 1; rows * rows <= size; ++rows) {
      if (size % rows == 0) {
        int cols = size / rows;
        int diff = std::abs(cols - rows);
        if (diff < best_diff) {
          best_diff = diff;
          grid_rows = rows;
          grid_cols = cols;
        }
      }
    }
  }

  int GetPixelClamped(const std::vector<uint8_t>& data, int x, int y, int width, int height, int channels, int channel) {
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    return data[(y * width + x) * channels + channel];
  }

  uint8_t ApplyGaussianFilter(const std::vector<uint8_t>& padded_data, int x, int y, 
                               int padded_width, int padded_height, int channels, int channel) {
    int sum = 0;
    
    for (int ky = -1; ky <= 1; ++ky) {
      for (int kx = -1; kx <= 1; ++kx) {
        int pixel = GetPixelClamped(padded_data, x + kx, y + ky, padded_width, padded_height, channels, channel);
        int k_value = kernel[(ky + 1) * 3 + (kx + 1)];
        sum += pixel * k_value;
      }
    }
    
    return static_cast<uint8_t>(sum / kernel_sum);
  }


}  // namespace

SmyshlaevAGaussFiltMPI::SmyshlaevAGaussFiltMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SmyshlaevAGaussFiltMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int error_flag = 0;
  if (rank == 0) {
    const InType &input_img = GetInput();
    if (input_img.width <= 0 || input_img.height <= 0 || input_img.channels <= 0 || input_img.data.empty()) {
      error_flag = 1;
    }
  }
  MPI_Bcast(&error_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return (error_flag == 0);
}

bool SmyshlaevAGaussFiltMPI::PreProcessingImpl() {
  return true;
}

bool SmyshlaevAGaussFiltMPI::RunImpl() {

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int grid_rows, grid_cols;
  FindOptimalGrid(size, grid_rows, grid_cols);

  InType input_image;
  if (rank == 0) {
    input_image = GetInput();
  }

  int img_width = input_image.width;
  int img_height = input_image.height;
  int img_channels = input_image.channels;

  MPI_Bcast(&img_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&img_height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&img_channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    input_image.width = img_width;
    input_image.height = img_height;
    input_image.channels = img_channels;
  }

  std::vector<BlockInfo> blocks(size);
  int block_height = (img_height + grid_rows - 1) / grid_rows;
  int block_width = (img_width + grid_cols - 1) / grid_cols;

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);
  std::vector<uint8_t> scatter_buffer;

  if (rank == 0) {
    int total_packed = 0;
    
    for (int p = 0; p < size; ++p) {
      int grid_row = p / grid_cols;
      int grid_col = p % grid_cols;

      int start_row = grid_row * block_height;
      int start_col = grid_col * block_width;

      int actual_block_height = std::min(block_height, img_height - start_row);
      int actual_block_width = std::min(block_width, img_width - start_col);

      int padded_top = (grid_row > 0) ? 1 : 0;
      int padded_bottom = (grid_row < grid_rows - 1) ? 1 : 0;
      int padded_left = (grid_col > 0) ? 1 : 0;
      int padded_right = (grid_col < grid_cols - 1) ? 1 : 0;

      int padded_height = actual_block_height + padded_top + padded_bottom;
      int padded_width = actual_block_width + padded_left + padded_right;

      blocks[p].start_row = start_row;
      blocks[p].start_col = start_col;
      blocks[p].block_height = actual_block_height;
      blocks[p].block_width = actual_block_width;
      blocks[p].padded_height = padded_height;
      blocks[p].padded_width = padded_width;
      blocks[p].count = padded_height * padded_width * img_channels;
      
      total_packed += blocks[p].count;
    }

    displs[0] = 0;
    sendcounts[0] = blocks[0].count;
    
    for (int p = 1; p < size; ++p) {
      displs[p] = displs[p - 1] + sendcounts[p - 1];
      sendcounts[p] = blocks[p].count;
    }

    scatter_buffer.resize(total_packed);

    for (int p = 0; p < size; ++p) {
      int grid_row = p / grid_cols;
      int grid_col = p % grid_cols;
      
      uint8_t* buffer_ptr = scatter_buffer.data() + displs[p];

      int padded_top = (grid_row > 0) ? 1 : 0;
      int padded_left = (grid_col > 0) ? 1 : 0;

      int top_row = blocks[p].start_row - padded_top;
      int left_col = blocks[p].start_col - padded_left;
      
      int end_row = std::min(img_height, blocks[p].start_row + blocks[p].block_height + 
                            ((grid_row < grid_rows - 1) ? 1 : 0));
      int end_col = std::min(img_width, blocks[p].start_col + blocks[p].block_width + 
                            ((grid_col < grid_cols - 1) ? 1 : 0));

      int padded_row = 0;
      for (int y = top_row; y < end_row; ++y) {
        int padded_col = 0;
        for (int x = left_col; x < end_col; ++x) {
          for (int ch = 0; ch < img_channels; ++ch) {
            buffer_ptr[(padded_row * blocks[p].padded_width + padded_col) * img_channels + ch] =
                input_image.data[(y * img_width + x) * img_channels + ch];
          }
          padded_col++;
        }
        padded_row++;
      }
    }
  }

  std::vector<int> block_info(size * 6);
  if (rank == 0) {
    for (int p = 0; p < size; ++p) {
      block_info[p * 6 + 0] = blocks[p].start_row;
      block_info[p * 6 + 1] = blocks[p].start_col;
      block_info[p * 6 + 2] = blocks[p].block_height;
      block_info[p * 6 + 3] = blocks[p].block_width;
      block_info[p * 6 + 4] = blocks[p].padded_height;
      block_info[p * 6 + 5] = blocks[p].padded_width;
    }
  }
  MPI_Bcast(block_info.data(), size * 6, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(sendcounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    blocks.clear();
    blocks.resize(size);
    for (int p = 0; p < size; ++p) {
      blocks[p].start_row = block_info[p * 6 + 0];
      blocks[p].start_col = block_info[p * 6 + 1];
      blocks[p].block_height = block_info[p * 6 + 2];
      blocks[p].block_width = block_info[p * 6 + 3];
      blocks[p].padded_height = block_info[p * 6 + 4];
      blocks[p].padded_width = block_info[p * 6 + 5];
      blocks[p].count = blocks[p].padded_height * blocks[p].padded_width * img_channels;
    }
  }

  std::vector<uint8_t> local_block_data(blocks[rank].count);
  MPI_Scatterv(rank == 0 ? scatter_buffer.data() : nullptr,
               rank == 0 ? sendcounts.data() : nullptr,
               rank == 0 ? displs.data() : nullptr,
               MPI_UNSIGNED_CHAR,
               local_block_data.data(),
               blocks[rank].count,
               MPI_UNSIGNED_CHAR,
               0,
               MPI_COMM_WORLD);

  int local_block_height = blocks[rank].block_height;
  int local_block_width = blocks[rank].block_width;
  int local_padded_height = blocks[rank].padded_height;
  int local_padded_width = blocks[rank].padded_width;

  int grid_row = rank / grid_cols;
  int grid_col = rank % grid_cols;
  int x_offset = (grid_col > 0) ? 1 : 0;
  int y_offset = (grid_row > 0) ? 1 : 0;

  std::vector<uint8_t> local_output_data(local_block_height * local_block_width * img_channels);

  for (int y = 0; y < local_block_height; ++y) {
    for (int x = 0; x < local_block_width; ++x) {
      for (int ch = 0; ch < img_channels; ++ch) {
        uint8_t filtered = ApplyGaussianFilter(local_block_data, x + x_offset, y + y_offset, 
                                               local_padded_width, local_padded_height,
                                               img_channels, ch);
        local_output_data[(y * local_block_width + x) * img_channels + ch] = filtered;
      }
    }
  }

  std::vector<int> recvcounts(size);
  std::vector<int> recv_displs(size, 0);

  if (rank == 0) {
    for (int p = 0; p < size; ++p) {
      recvcounts[p] = blocks[p].block_height * blocks[p].block_width * img_channels;
      if (p > 0) {
        recv_displs[p] = recv_displs[p - 1] + recvcounts[p - 1];
      }
    }
  }

  std::vector<uint8_t> gathered_data;
  if (rank == 0) {
    gathered_data.resize(img_width * img_height * img_channels);
  }

  MPI_Gatherv(local_output_data.data(),
              blocks[rank].block_height * blocks[rank].block_width * img_channels,
              MPI_UNSIGNED_CHAR,
              gathered_data.data(),
              rank == 0 ? recvcounts.data() : nullptr,
              rank == 0 ? recv_displs.data() : nullptr,
              MPI_UNSIGNED_CHAR,
              0,
              MPI_COMM_WORLD);

  if (rank == 0) {
    OutType& output_image = GetOutput();
    output_image.width = img_width;
    output_image.height = img_height;
    output_image.channels = img_channels;
    output_image.data.resize(img_width * img_height * img_channels);

    for (int p = 0; p < size; ++p) {
      int start_row = blocks[p].start_row;
      int start_col = blocks[p].start_col;

      uint8_t* src = gathered_data.data() + recv_displs[p];

      for (int y = 0; y < blocks[p].block_height; ++y) {
        for (int x = 0; x < blocks[p].block_width; ++x) {
          int global_y = start_row + y;
          int global_x = start_col + x;
          for (int ch = 0; ch < img_channels; ++ch) {
            output_image.data[(global_y * img_width + global_x) * img_channels + ch] =
                src[(y * blocks[p].block_width + x) * img_channels + ch];
          }
        }
      }
    }
  }

  return true;
}

bool SmyshlaevAGaussFiltMPI::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_gauss_filt
