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

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&kernel_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return width > 0 && height > 0 && kernel_size > 0;
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

  MPI_Bcast(&GetOutput().width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&GetOutput().height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&GetOutput().kernel_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

namespace {
void ProcessLocalRows(int start_row, int end_row, int width, int height, int kernel_size,
                      const std::vector<uint8_t> &input, std::vector<uint8_t> &output) {
  const int radius = kernel_size / 2;

  for (int row_idx = start_row; row_idx < end_row; ++row_idx) {
    for (int col_idx = 0; col_idx < width; ++col_idx) {
      int sum = 0;
      int count = 0;
      for (int kernel_y = -radius; kernel_y <= radius; ++kernel_y) {
        for (int kernel_x = -radius; kernel_x <= radius; ++kernel_x) {
          const int ny = Clamp(row_idx + kernel_y, 0, height - 1);
          const int nx = Clamp(col_idx + kernel_x, 0, width - 1);
          sum += input[(ny * width) + nx];
          ++count;
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

  const int width = GetOutput().width;
  const int height = GetOutput().height;
  const int kernel_size = GetOutput().kernel_size;

  std::vector<uint8_t> full_input_data;
  if (rank == 0) {
    full_input_data = GetInput().data;
  } else {
    full_input_data.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
  }

  MPI_Bcast(full_input_data.data(), width * height, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  const int delta = height / size;
  const int remainder = height % size;
  const int start_row = (rank * delta) + std::min(rank, remainder);
  const int end_row = start_row + delta + (rank < remainder ? 1 : 0);
  const int local_rows = end_row - start_row;

  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);
  for (int i = 0; i < size; ++i) {
    const int r_start = (i * delta) + std::min(i, remainder);
    const int r_end = r_start + delta + (i < remainder ? 1 : 0);
    recv_counts[i] = (r_end - r_start) * width;
    displs[i] = r_start * width;
  }

  if (local_rows > 0) {
    std::vector<uint8_t> local_result(static_cast<size_t>(local_rows) * static_cast<size_t>(width));
    ProcessLocalRows(start_row, end_row, width, height, kernel_size, full_input_data, local_result);

    MPI_Gatherv(local_result.data(), local_rows * width, MPI_UNSIGNED_CHAR,
                (rank == 0 ? GetOutput().data.data() : nullptr), recv_counts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);
  } else {
    MPI_Gatherv(nullptr, 0, MPI_UNSIGNED_CHAR, (rank == 0 ? GetOutput().data.data() : nullptr), recv_counts.data(),
                displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool GaseninLImageSmoothMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gasenin_l_image_smooth
