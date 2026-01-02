#include "kiselev_i_linear_histogram_stretch\mpi\include\ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <limits>
#include <vector>

#include "kiselev_i_linear_histogram_stretch/common/include/common.hpp"

namespace kiselev_i_linear_histogram_stretch {

KiselevITestTaskMPI::KiselevITestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;

  if (!in.pixels.empty()) {
    GetOutput().resize(in.pixels.size());
  }
}

bool KiselevITestTaskMPI::ValidationImpl() {
  const auto &img = GetInput();
  return img.width > 0 && img.height > 0 && img.pixels.size() == img.width * img.height;
}

bool KiselevITestTaskMPI::PreProcessingImpl() {
  return true;
}

bool KiselevITestTaskMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int root = 0;

  std::size_t total_size = 0;
  if (rank == root) {
    total_size = GetInput().pixels.size();
  }

  // Рассылка общего количества элементов
  MPI_Bcast(&total_size, 1, MPI_UNSIGNED_LONG_LONG, root, MPI_COMM_WORLD);

  const std::size_t base = total_size / size;
  const std::size_t extra = total_size % size;

  std::vector<int> counts(size, 0);
  std::vector<int> offsets(size, 0);

  if (rank == root) {
    std::size_t shift = 0;
    for (int i = 0; i < size; ++i) {
      counts[i] = static_cast<int>(base + (i < static_cast<int>(extra) ? 1 : 0));
      offsets[i] = static_cast<int>(shift);
      shift += counts[i];
    }
  }

  int local_count = 0;
  MPI_Scatter(counts.data(), 1, MPI_INT, &local_count, 1, MPI_INT, root, MPI_COMM_WORLD);

  std::vector<uint8_t> local_pixels(local_count);

  MPI_Scatterv(GetInput().pixels.data(), counts.data(), offsets.data(), MPI_UNSIGNED_CHAR, local_pixels.data(),
               local_count, MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);

  uint8_t local_min = std::numeric_limits<uint8_t>::max();
  uint8_t local_max = std::numeric_limits<uint8_t>::min();

  if (local_pixels.empty()) {
    local_min = std::numeric_limits<uint8_t>::max();
    local_max = std::numeric_limits<uint8_t>::min();
  }

  for (uint8_t px : local_pixels) {
    if (px < local_min) {
      local_min = px;
    }
    if (px > local_max) {
      local_max = px;
    }
  }

  uint8_t global_min = 0;
  uint8_t global_max = 0;

  MPI_Allreduce(&local_min, &global_min, 1, MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, &global_max, 1, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);

  if (global_min != global_max) {
    const double scale = 255.0 / static_cast<double>(global_max - global_min);

    for (std::size_t i = 0; i < local_pixels.size(); ++i) {
      const double value = static_cast<double>(local_pixels[i] - global_min) * scale;
      local_pixels[i] = static_cast<uint8_t>(value + 0.5);
    }
  }

  MPI_Gatherv(local_pixels.data(), local_count, MPI_UNSIGNED_CHAR, GetOutput().data(), counts.data(), offsets.data(),
              MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);
  return true;
}

bool KiselevITestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kiselev_i_linear_histogram_stretch
