#include "egorova_l_gauss_filter_vert/mpi/include/ops_mpi.hpp"
#include <mpi.h>
#include <algorithm>

namespace egorova_l_gauss_filter_vert {

EgorovaLGaussFilterVertMPI::EgorovaLGaussFilterVertMPI(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;  // Используем GetInput() для доступа к input_
}

bool EgorovaLGaussFilterVertMPI::ValidationImpl() {
  auto& input = GetInput();  // Получаем ссылку на входные данные
  return input.rows >= 3 && input.cols >= 3;
}

bool EgorovaLGaussFilterVertMPI::PreProcessingImpl() { return true; }

bool EgorovaLGaussFilterVertMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto& full_img = GetInput();  // Используем GetInput()
  auto& output = GetOutput();   // Используем GetOutput()
  
  int r = full_img.rows;
  int c = full_img.cols;
  int ch = full_img.channels;

  int q = c / size;
  int rem = c % size;
  int local_cols = q + (rank < rem ? 1 : 0);
  int offset = rank * q + std::min(rank, rem);

  int left_h = (rank > 0) ? 1 : 0;
  int right_h = (rank < size - 1) ? 1 : 0;
  int total_lc = local_cols + left_h + right_h;

  std::vector<uint8_t> local_in(r * total_lc * ch);
  std::vector<uint8_t> local_out(r * local_cols * ch);

  for (int i = 0; i < r; ++i) {
    std::copy(full_img.data.begin() + (i * c + offset - left_h) * ch,
              full_img.data.begin() + (i * c + offset + local_cols + right_h) * ch,
              local_in.begin() + (i * total_lc * ch));
  }

  const float ker[3][3] = {{0.0625f, 0.125f, 0.0625f}, {0.125f, 0.25f, 0.125f}, {0.0625f, 0.125f, 0.0625f}};

  for (int y = 0; y < r; ++y) {
    for (int x = 0; x < local_cols; ++x) {
      int in_x = x + left_h;
      for (int k = 0; k < ch; ++k) {
        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            int py = std::clamp(y + ky, 0, r - 1);
            int px = in_x + kx;
            sum += local_in[(py * total_lc + px) * ch + k] * ker[ky + 1][kx + 1];
          }
        }
        local_out[(y * local_cols + x) * ch + k] = static_cast<uint8_t>(sum);
      }
    }
  }

  std::vector<int> counts(size), displs(size);
  for (int i = 0; i < size; ++i) {
    counts[i] = (q + (i < rem ? 1 : 0)) * ch;
    displs[i] = (i * q + std::min(i, rem)) * ch;
  }

  // Создаем output и копируем в него результат
  output.rows = r;
  output.cols = c;
  output.channels = ch;
  output.data.resize(r * c * ch);
  
  for (int i = 0; i < r; ++i) {
    MPI_Gatherv(local_out.data() + i * local_cols * ch, local_cols * ch, MPI_BYTE,
                output.data.data() + i * c * ch, counts.data(), displs.data(), 
                MPI_BYTE, 0, MPI_COMM_WORLD);
  }
  
  // Для процесса с рангом 0 результат готов, для других процессов output пуст
  // Это нормально для MPI реализации
  
  return true;
}

bool EgorovaLGaussFilterVertMPI::PostProcessingImpl() { return true; }
}