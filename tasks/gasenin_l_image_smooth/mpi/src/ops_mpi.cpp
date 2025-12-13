#include "gasenin_l_image_smooth/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace gasenin_l_image_smooth {

GaseninLImageSmoothMPI::GaseninLImageSmoothMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GaseninLImageSmoothMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int width = GetInput().width;
  int height = GetInput().height;

  // Рассылаем размеры всем, чтобы все процессы прошли валидацию одинаково
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return width > 0 && height > 0;
}

bool GaseninLImageSmoothMPI::PreProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() = GetInput();
    GetOutput().data.assign(GetInput().data.size(), 0);
  } else {
    // Не-root процессам тоже нужны метаданные в GetOutput для корректной работы RunImpl
    GetOutput().width = GetInput().width;
    GetOutput().height = GetInput().height;
    GetOutput().kernel_size = GetInput().kernel_size;
  }

  // Синхронизируем метаданные вывода
  MPI_Bcast(&GetOutput().width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&GetOutput().height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&GetOutput().kernel_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool GaseninLImageSmoothMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int w = GetOutput().width;
  int h = GetOutput().height;
  int k_size = GetOutput().kernel_size;

  std::vector<uint8_t> full_input_data;
  if (rank == 0) {
    full_input_data = GetInput().data;
  } else {
    full_input_data.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
  }

  // Рассылаем всё изображение всем процессам
  MPI_Bcast(full_input_data.data(), w * h, MPI_UINT8_T, 0, MPI_COMM_WORLD);

  // Расчет строк
  int delta = h / size;
  int rem = h % size;
  int start_row = rank * delta + std::min(rank, rem);
  int end_row = start_row + delta + (rank < rem ? 1 : 0);
  int local_rows = end_row - start_row;

  if (local_rows > 0) {
    std::vector<uint8_t> local_result(local_rows * w);
    int radius = k_size / 2;

    for (int y = start_row; y < end_row; ++y) {
      for (int x = 0; x < w; ++x) {
        int sum = 0;
        int count = 0;
        for (int ky = -radius; ky <= radius; ++ky) {
          for (int kx = -radius; kx <= radius; ++kx) {
            int ny = clamp(y + ky, 0, h - 1);
            int nx = clamp(x + kx, 0, w - 1);
            sum += full_input_data[ny * w + nx];
            count++;
          }
        }
        local_result[(y - start_row) * w + x] = static_cast<uint8_t>(sum / count);
      }
    }

    // Подготовка к сборке
    std::vector<int> recv_counts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; ++i) {
      int r_start = i * delta + std::min(i, rem);
      int r_end = r_start + delta + (i < rem ? 1 : 0);
      recv_counts[i] = (r_end - r_start) * w;
      displs[i] = r_start * w;
    }

    MPI_Gatherv(local_result.data(), local_rows * w, MPI_UINT8_T, (rank == 0 ? GetOutput().data.data() : nullptr),
                recv_counts.data(), displs.data(), MPI_UINT8_T, 0, MPI_COMM_WORLD);
  } else {
    // Процессы без строк тоже должны участвовать в коллективном Gatherv
    std::vector<int> recv_counts(size, 0);
    std::vector<int> displs(size, 0);
    if (rank == 0) { /* заполнить counts */
    }
    MPI_Gatherv(nullptr, 0, MPI_UINT8_T, nullptr, nullptr, nullptr, MPI_UINT8_T, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool GaseninLImageSmoothMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gasenin_l_image_smooth
