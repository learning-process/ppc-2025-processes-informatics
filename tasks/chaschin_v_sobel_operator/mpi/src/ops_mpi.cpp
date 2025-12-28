#include "chaschin_v_sobel_operator/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "chaschin_v_sobel_operator/common/include/common.hpp"

namespace chaschin_v_sobel_operator {

ChaschinVSobelOperatorMPI::ChaschinVSobelOperatorMPI(const InType &in) {
  GetInput() = in;
  GetOutput().clear();

  const auto &image = std::get<0>(in);
  int height = std::get<1>(in);
  int width = std::get<2>(in);

  Size = std::make_tuple(height, width);
}

bool ChaschinVSobelOperatorMPI::ValidationImpl() {
  const auto &in = GetInput();

  const auto &image = std::get<0>(in);

  return !image.empty();
}

std::vector<float> PreprocessToGrayscaleWithOverlap(
    const std::vector<std::vector<Pixel>> &image, int n_procs,
    std::vector<int> &sendcounts,  // выход: количество элементов для каждого процесса
    std::vector<int> &displs       // выход: смещения в массиве
) {
  int n = image.size();
  int m = image[0].size();

  int padded_m = m + 2;

  // Расчет блоков для каждого процесса
  sendcounts.resize(n_procs);
  displs.resize(n_procs);

  int base = n / n_procs;
  int rem = n % n_procs;

  // Считаем общее количество строк с «верхней и нижней» для scatter
  int total_rows = 0;
  for (int rank = 0; rank < n_procs; ++rank) {
    int local_rows = base + (rank < rem ? 1 : 0);
    sendcounts[rank] = (local_rows + 2) * padded_m;  // +2 — верхняя и нижняя строки
    displs[rank] = total_rows * padded_m;
    total_rows += local_rows;  // только реальные строки, без двойного учета паддинга
  }

  // Общий одномерный буфер для scatterv
  std::vector<float> buffer(total_rows * padded_m + 2 * padded_m, 0.0f);  // добавим верх/низ для первого/последнего

  // Заполняем buffer с padding и overlap
  for (int rank = 0; rank < n_procs; ++rank) {
    int local_rows = base + (rank < rem ? 1 : 0);
    int offset = base * rank + std::min(rank, rem);

    // Буфер начала блока в общем массиве
    int start_idx = displs[rank];

    // Верхняя строка: если rank==0, нулевая, иначе повторяем последнюю строку предыдущего блока
    if (rank == 0) {
      // уже 0
    } else {
      for (int j = 0; j < m; ++j) {
        buffer[start_idx + j + 1] = 0;  // или можно скопировать реальные значения предыдущей строки
      }
    }

    // Центральные строки
    for (int i = 0; i < local_rows; ++i) {
      for (int j = 0; j < m; ++j) {
        const Pixel &p = image[offset + i][j];
        float val = 0.299f * p.r + 0.587f * p.g + 0.114f * p.b;
        buffer[start_idx + (i + 1) * padded_m + (j + 1)] = val;
      }
    }

    // Нижняя строка: если rank==n_procs-1, нулевая, иначе повторяем первую строку следующего блока
    if (rank == n_procs - 1) {
      // уже 0
    }
  }

  return buffer;
}

bool ChaschinVSobelOperatorMPI::PreProcessingImpl() {
  const auto &in = GetInput();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Вывод размеров изображения
  std::cout << "Process " << rank << " get image " << std::get<1>(in) << "x" << std::get<2>(in) << std::endl;

  if (rank == 0) {
    const auto &image = std::get<0>(in);
    int n_procs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    // Векторы для Scatterv
    std::vector<int> sendcounts;
    std::vector<int> displs;

    // Препроцессинг: формируем одномерный массив с padding и overlap для scatterv
    PreProcessGray = PreprocessToGrayscaleWithOverlap(image, n_procs, sendcounts, displs);

    // Можно сохранить sendcounts и displs для RunImpl, если нужно
    ScatterSendCounts = sendcounts;
    ScatterDispls = displs;
  }

  return true;
}

float sobel_at(const std::vector<float> &img, int i, int j, int stride) {
  static const int Kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  static const int Ky[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  float gx = 0.0f, gy = 0.0f;

  // i,j — индексы в массиве с padding
  for (int di = -1; di <= 1; ++di) {
    int ni = i + di;
    for (int dj = -1; dj <= 1; ++dj) {
      int nj = j + dj;
      float val = img[ni * stride + nj];  // одномерная индексация
      gx += val * Kx[di + 1][dj + 1];
      gy += val * Ky[di + 1][dj + 1];
    }
  }

  return std::sqrt(gx * gx + gy * gy);
}

bool ChaschinVSobelOperatorMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = std::get<0>(Size);
  int m = std::get<1>(Size);

  // sendcounts и displs уже подготовлены в PreProcessingImpl
  const auto &in = PreProcessGray;

  // Локальные параметры
  int base = n / size;
  int rem = n % size;
  int local_rows = base + (rank < rem ? 1 : 0);
  int padded_m = m + 2;

  // Локальный буфер с padding сверху и снизу (все уже подготовлено)
  std::vector<float> local_block(local_rows * padded_m);

  // Scatterv центральных строк (вверх/низ уже в PreProcessGray)
  MPI_Scatterv(rank == 0 ? in.data() : nullptr, ScatterSendCounts.data(), ScatterDispls.data(), MPI_FLOAT,
               local_block.data(), local_rows * padded_m, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // --- Локальный Sobel ---
  std::vector<float> local_output(local_rows * m);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < local_rows; ++i) {
    for (int j = 0; j < m; ++j) {
      // local_block уже содержит padding, поэтому смещаем на +1 строки и +1 столбцы
      local_output[i * m + j] = sobel_at(local_block, i + 1, j + 1, padded_m);
    }
  }

  // --- Сбор результатов ---
  std::vector<int> recvcounts(size), displs_out(size);
  int offset_res = 0;
  for (int p = 0; p < size; ++p) {
    int rows_p = base + (p < rem ? 1 : 0);
    recvcounts[p] = rows_p * m;
    displs_out[p] = offset_res;
    offset_res += recvcounts[p];
  }

  if (rank == 0) {
    PostProcessGray.resize(n * m);
  }

  MPI_Gatherv(local_output.data(), local_rows * m, MPI_FLOAT, rank == 0 ? PostProcessGray.data() : nullptr,
              recvcounts.data(), displs_out.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

  return true;
}

bool ChaschinVSobelOperatorMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Только rank 0 имеет данные
  if (rank != 0) {
    return true;
  }

  auto &out = GetOutput();

  int n = std::get<0>(Size);
  int m = std::get<1>(Size);

  if (static_cast<int>(PostProcessGray.size()) != n * m) {
    throw std::runtime_error("PostProcessingImpl: size mismatch");
  }

  out.resize(n);
  float max_val = 0.0f;

  // Найдем максимум для нормализации
  for (float v : PostProcessGray) {
    max_val = std::max(max_val, v);
  }

  float inv = (max_val > 0.0f) ? (255.0f / max_val) : 0.0f;

  for (int i = 0; i < n; ++i) {
    out[i].resize(m);
    for (int j = 0; j < m; ++j) {
      float v = PostProcessGray[i * m + j] * inv;
      unsigned char c = static_cast<unsigned char>(std::clamp(v, 0.0f, 255.0f));
      out[i][j] = Pixel{c, c, c};
    }
  }

  return true;
}

}  // namespace chaschin_v_sobel_operator
