#include "makovskiy_i_gauss_filter_vert/mpi/include/ops_mpi.hpp"
#include <mpi.h>
#include <algorithm>
#include <vector>

namespace makovskiy_i_gauss_filter_vert {

GaussFilterMPI::GaussFilterMPI(const InType& in) {
  InType temp(in);
  this->GetInput().swap(temp);
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool GaussFilterMPI::ValidationImpl() {
  const auto& [input, width, height] = GetInput();
  return !input.empty() && width > 0 && height > 0 && input.size() == static_cast<size_t>(width * height);
}

bool GaussFilterMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto& [input, width, height] = GetInput();
    total_width = width;
    total_height = height;
  }

  MPI_Bcast(&total_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&total_height, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int min_strip_width = total_width / world_size;
  const int remainder = total_width % world_size;
  strip_width = min_strip_width + (rank < remainder ? 1 : 0);

  if (strip_width > 0) {
      local_strip.resize(strip_width * total_height);
  }

  if (rank == 0) {
      GetOutput().resize(total_width * total_height);
  }
  
  return true;
}

bool GaussFilterMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto& [input, width, height] = GetInput();
  auto& final_output = GetOutput();

  if (rank == 0) {
      int offset = 0;
      for (int i = 0; i < world_size; ++i) {
          int current_strip_width = total_width / world_size + (i < total_width % world_size ? 1 : 0);
          if (current_strip_width == 0) continue;

          std::vector<int> strip_to_send(current_strip_width * total_height);
          for (int y = 0; y < total_height; ++y) {
              for (int x = 0; x < current_strip_width; ++x) {
                  strip_to_send[y * current_strip_width + x] = input[y * total_width + offset + x];
              }
          }

          if (i == 0) {
              local_strip = strip_to_send;
          } else {
              MPI_Send(strip_to_send.data(), strip_to_send.size(), MPI_INT, i, 0, MPI_COMM_WORLD);
          }
          offset += current_strip_width;
      }
  } else {
      if (strip_width > 0) {
          MPI_Recv(local_strip.data(), local_strip.size(), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
  }

  std::vector<int> local_output;
  if (strip_width > 0) {
    std::vector<int> left_ghost(total_height);
    std::vector<int> right_ghost(total_height);
    MPI_Request requests[4];
    int req_count = 0;

    std::vector<int> left_border(total_height);
    std::vector<int> right_border(total_height);

    for (int y = 0; y < total_height; ++y) {
        left_border[y] = local_strip[y * strip_width];
        right_border[y] = local_strip[y * strip_width + strip_width - 1];
    }
    
    std::vector<int> all_strip_widths(world_size);
    MPI_Allgather(&strip_width, 1, MPI_INT, all_strip_widths.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    if (rank > 0 && all_strip_widths[rank - 1] > 0) {
      MPI_Isend(left_border.data(), total_height, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
      MPI_Irecv(left_ghost.data(), total_height, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }
    if (rank < world_size - 1 && all_strip_widths[rank + 1] > 0) {
      MPI_Isend(right_border.data(), total_height, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
      MPI_Irecv(right_ghost.data(), total_height, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

    local_output.resize(strip_width * total_height);
    const int kernel[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    const int kernel_sum = 16;
    
    for (int y = 0; y < total_height; ++y) {
      for (int x = 0; x < strip_width; ++x) {
        int sum = 0;
        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            int pixel_val;
            int current_x = x + kx;
            int current_y = y + ky;
            
            if (current_x < 0) {
                pixel_val = (rank > 0 && all_strip_widths[rank - 1] > 0) ? left_ghost[std::clamp(current_y, 0, total_height - 1)] : get_pixel(local_strip, current_x, current_y, strip_width, total_height);
            } else if (current_x >= strip_width) {
                pixel_val = (rank < world_size - 1 && all_strip_widths[rank + 1] > 0) ? right_ghost[std::clamp(current_y, 0, total_height - 1)] : get_pixel(local_strip, current_x, current_y, strip_width, total_height);
            } else {
                pixel_val = get_pixel(local_strip, current_x, current_y, strip_width, total_height);
            }
            sum += pixel_val * kernel[(ky + 1) * 3 + (kx + 1)];
          }
        }
        local_output[y * strip_width + x] = sum / kernel_sum;
      }
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
      int offset = 0;
      for (int i = 0; i < world_size; ++i) {
          int current_strip_width = total_width / world_size + (i < total_width % world_size ? 1 : 0);
          if (current_strip_width == 0) continue;

          std::vector<int> received_strip(current_strip_width * total_height);
          if (i == 0) {
              received_strip = local_output;
          } else {
              MPI_Recv(received_strip.data(), received_strip.size(), MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }

          for (int y = 0; y < total_height; ++y) {
              for (int x = 0; x < current_strip_width; ++x) {
                  final_output[y * total_width + offset + x] = received_strip[y * current_strip_width + x];
              }
          }
          offset += current_strip_width;
      }
  } else {
      if (strip_width > 0) {
          MPI_Send(local_output.data(), local_output.size(), MPI_INT, 0, 1, MPI_COMM_WORLD);
      }
  }

  return true;
}

bool GaussFilterMPI::PostProcessingImpl() { return true; }

}  // namespace makovskiy_i_gauss_filter_vert