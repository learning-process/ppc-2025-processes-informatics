#include "makovskiy_i_gauss_filter_vert/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>

namespace makovskiy_i_gauss_filter_vert {

GaussFilterMPI::GaussFilterMPI(const InType &in) {
  InType temp(in);
  this->GetInput().swap(temp);
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool GaussFilterMPI::ValidationImpl() {  // NOLINT(readability-convert-member-functions-to-static)
  const auto &[input, width, height] = GetInput();
  return !input.empty() && width > 0 && height > 0 && input.size() == static_cast<size_t>(width * height);
}

bool GaussFilterMPI::PreProcessingImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (rank == 0) {
    const auto &[input, width, height] = GetInput();
    total_width_ = width;
    total_height_ = height;
  }

  MPI_Bcast(&total_width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&total_height_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  strip_width_ = (total_width_ / world_size) + (rank < total_width_ % world_size ? 1 : 0);

  if (strip_width_ > 0) {
    local_strip_.resize(strip_width_ * total_height_);
  }

  return true;
}

bool GaussFilterMPI::RunImpl() {  // NOLINT(readability-function-cognitive-complexity)
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<int> all_strip_widths(world_size);
  MPI_Allgather(&strip_width_, 1, MPI_INT, all_strip_widths.data(), 1, MPI_INT, MPI_COMM_WORLD);

  if (rank == 0) {
    const auto &[input, width, height] = GetInput();
    int offset = 0;
    for (int proc = 0; proc < world_size; ++proc) {
      int current_strip_width = (total_width_ / world_size) + (proc < total_width_ % world_size ? 1 : 0);
      if (current_strip_width == 0) {
        continue;
      }

      std::vector<int> strip_to_send(current_strip_width * total_height_);
      for (int y_coord = 0; y_coord < total_height_; ++y_coord) {
        for (int x_coord = 0; x_coord < current_strip_width; ++x_coord) {
          strip_to_send[(y_coord * current_strip_width) + x_coord] = input[(y_coord * total_width_) + offset + x_coord];
        }
      }

      if (proc == 0) {
        local_strip_ = strip_to_send;
      } else {
        MPI_Send(strip_to_send.data(), strip_to_send.size(), MPI_INT, proc, 0, MPI_COMM_WORLD);
      }
      offset += current_strip_width;
    }
  } else {
    if (strip_width_ > 0) {
      MPI_Recv(local_strip_.data(), local_strip_.size(), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  if (strip_width_ > 0) {
    std::vector<int> left_ghost(total_height_);
    std::vector<int> right_ghost(total_height_);
    std::vector<int> left_border(total_height_);
    std::vector<int> right_border(total_height_);

    std::array<MPI_Request, 4> requests{};
    int req_count = 0;

    for (int y_coord = 0; y_coord < total_height_; ++y_coord) {
      left_border[y_coord] = local_strip_[y_coord * strip_width_];
      right_border[y_coord] = local_strip_[(y_coord * strip_width_) + strip_width_ - 1];
    }

    if (rank > 0 && all_strip_widths[rank - 1] > 0) {
      MPI_Isend(left_border.data(), total_height_, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
      MPI_Irecv(left_ghost.data(), total_height_, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }
    if (rank < world_size - 1 && all_strip_widths[rank + 1] > 0) {
      MPI_Isend(right_border.data(), total_height_, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
      MPI_Irecv(right_ghost.data(), total_height_, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }

    MPI_Waitall(req_count, requests.data(), MPI_STATUSES_IGNORE);

    constexpr std::array<int, 9> kKernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    constexpr int kKernelSum = 16;
    std::vector<int> local_output(strip_width_ * total_height_);

    for (int y_coord = 0; y_coord < total_height_; ++y_coord) {
      for (int x_coord = 0; x_coord < strip_width_; ++x_coord) {
        int sum = 0;
        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            int current_x = x_coord + kx;
            int pixel_val = 0;

            bool is_left_ghost = current_x < 0 && rank > 0 && all_strip_widths[rank - 1] > 0;
            bool is_right_ghost = current_x >= strip_width_ && rank < world_size - 1 && all_strip_widths[rank + 1] > 0;

            if (is_left_ghost || is_right_ghost) {
              const auto &ghost_source = is_left_ghost ? left_ghost : right_ghost;
              pixel_val = GetPixel(ghost_source, 0, y_coord + ky, 1, total_height_);
            } else {
              int clamped_x = std::clamp(current_x, 0, strip_width_ - 1);
              pixel_val = GetPixel(local_strip_, clamped_x, y_coord + ky, strip_width_, total_height_);
            }

            sum += pixel_val * kKernel[static_cast<size_t>(((ky + 1) * 3) + (kx + 1))];
          }
        }
        local_output[(y_coord * strip_width_) + x_coord] = sum / kKernelSum;
      }
    }
    local_strip_ = std::move(local_output);
  }

  if (rank == 0) {
    std::vector<int> final_output(total_width_ * total_height_);
    int offset = 0;
    for (int proc = 0; proc < world_size; ++proc) {
      int current_strip_width = (total_width_ / world_size) + (proc < total_width_ % world_size ? 1 : 0);
      if (current_strip_width == 0) {
        continue;
      }

      std::vector<int> received_strip;
      if (proc == 0) {
        received_strip = local_strip_;
      } else {
        received_strip.resize(current_strip_width * total_height_);
        MPI_Recv(received_strip.data(), received_strip.size(), MPI_INT, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      for (int y_coord = 0; y_coord < total_height_; ++y_coord) {
        for (int x_coord = 0; x_coord < current_strip_width; ++x_coord) {
          final_output[(y_coord * total_width_) + offset + x_coord] =
              received_strip[(y_coord * current_strip_width) + x_coord];
        }
      }
      offset += current_strip_width;
    }
    GetOutput() = std::move(final_output);
  } else {
    if (strip_width_ > 0) {
      MPI_Send(local_strip_.data(), local_strip_.size(), MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
  }

  return true;
}

bool GaussFilterMPI::PostProcessingImpl() {  // NOLINT(readability-convert-member-functions-to-static)
  return true;
}

}  // namespace makovskiy_i_gauss_filter_vert
