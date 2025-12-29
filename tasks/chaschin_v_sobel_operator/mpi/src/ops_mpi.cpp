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
  auto in_copy = in;

  GetInput() = std::move(in_copy);
  GetOutput().clear();

  int height = std::get<1>(in);
  int width = std::get<2>(in);

  Size = std::make_tuple(height, width);
}

bool ChaschinVSobelOperatorMPI::ValidationImpl() {
  const auto &in = GetInput();

  const auto &image = std::get<0>(in);

  return !image.empty();
}

std::vector<float> ChaschinVSobelOperatorMPI::PreprocessToGrayscaleWithOverlap(
    const std::vector<std::vector<Pixel>> &image, int n_procs, std::vector<int> &sendcounts, std::vector<int> &displs) {
  int n = image.size();     // количество строк
  int m = image[0].size();  // количество столбцов

  int padded_m = m + 2;  // padding слева и справа

  // --- Расчет блоков для каждого процесса ---
  sendcounts.resize(n_procs);
  displs.resize(n_procs);

  int base = n / n_procs;
  int rem = n % n_procs;

  int total_real_rows = 0;  // без паддинга
  for (int rank = 0; rank < n_procs; ++rank) {
    int local_rows = (base + 2) + (rank < rem ? 1 : 0);
    sendcounts[rank] = (local_rows)*padded_m;   // +2 для верхней и нижней строк
    displs[rank] = total_real_rows * padded_m;  // смещение относительно первых реальных строк
    total_real_rows += local_rows;
  }

  // --- Создаем общий буфер с нулевым паддингом сверху/снизу ---
  std::vector<float> buffer((n + 2 + (n_procs - 1) * 2) * padded_m, 0.0f);

  std::vector<int> l_r(n_procs + 1, 0);

  for (int rank = 0; rank < n_procs + 1; ++rank) {
    l_r[rank + 1] = l_r[rank] + (base) + (rank < rem ? 1 : 0);
  }

  // std::cout<<"Prep:\n";
  //  --- Заполняем центральные строки ---
  for (int i = 0; i < n_procs; i++) {
    for (int x = -1; x < sendcounts[i] / padded_m - 1; x++) {
      for (int y = 0; y < m; y++) {
        float val;
        if (x + l_r[i] < 0 || x + l_r[i] >= n) {
          val = 0.0f;
        } else {
          const Pixel &p = image[x + l_r[i]][y];
          val = 0.299f * p.r + 0.587f * p.g + 0.114f * p.b;
        }
        buffer[displs[i] + (x + 1) * padded_m + y + 1] = val;
        // std::cout<<val<<" "<<i<<" "<<x+l_r[i]<<" "<<y<<" "<<displs[i] + (x+1) * padded_m + y + 1<<" \n";
      }
      // std::cout<<"\n";
    }
  }

  // --- Добавляем overlap для MPI (повтор верхней и нижней строки блока) ---
  // Сделаем это в Scatterv каждый процесс будет брать свои строки, включая верх/низ для свёртки

  return buffer;
}

bool ChaschinVSobelOperatorMPI::PreProcessingImpl() {
  const auto &in = GetInput();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Вывод размеров изображения
  // std::cout << "Process " << rank << " get image " << std::get<1>(in) << "x" << std::get<2>(in) << std::endl;

  if (rank == 0) {
    const auto &image = std::get<0>(in);
    /*
    for (int i = 0; i < std::get<1>(in); i++) {
      for (int j = 0; j < std::get<2>(in); j++) {
        std::cout<< "{" <<
    static_cast<int>(image[i][j].r)<<","<<static_cast<int>(image[i][j].g)<<","<<static_cast<int>(image[i][j].b)<< "}";
      }
      std::cout << "\n";
    }
    std::cout << "\n";*/

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

float ChaschinVSobelOperatorMPI::sobel_at(const std::vector<float> &img, int i, int j, int stride) {
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

  std::cout << "sendcounts и displs уже подготовлены в PreProcessingImpl\n" << std::flush;
  const auto &in = PreProcessGray;
  if (rank == 0) {
    for (int i = 0; i < n + 2 + (size - 1) * 2; i++) {
      for (int j = 0; j < m + 2; j++) {
        std::cout << PreProcessGray[i * (m + 2) + j] << " " << std::flush;
      }
      std::cout << "\n" << std::flush;
    }
    std::cout << "\n" << std::flush;
  }

  std::cout << "Локальные параметры\n" << std::flush;
  int base = n / size;
  int rem = n % size;
  int local_rows = (base + 2) + (rank < rem ? 1 : 0);
  int padded_m = m + 2;

  std::cout << "Локальный буфер с padding сверху и снизу (все уже подготовлено)\n" << std::flush;
  std::vector<float> local_block(local_rows * padded_m);

  std::cout << "Scatterv центральных строк (вверх/низ уже в PreProcessGray)\n" << std::flush;
  MPI_Scatterv(rank == 0 ? in.data() : nullptr, ScatterSendCounts.data(), ScatterDispls.data(), MPI_FLOAT,
               local_block.data(), (local_rows)*padded_m, MPI_FLOAT, 0, MPI_COMM_WORLD);

  std::cout << " --- Локальный Sobel ---\n" << std::flush;
  std::vector<float> local_output(local_rows * m);

  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < local_rows; ++i) {
    for (int j = 0; j < m; ++j) {
      // local_block уже содержит padding, поэтому смещаем на +1 строки и +1 столбцы
      local_output[i * m + j] = sobel_at(local_block, i + 1, j + 1, padded_m);

      // std::cout << "Process " << rank << " local_output " << local_output[i * m + j] << std::endl;
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

  /*for (int i = 0; i < local_output.size(); i++){
    std::cout<<rank<<":"<<local_output[i]<<" ";
  }
  std::cout<<"\n";*/

  MPI_Gatherv(local_output.data(), (local_rows - 2) * m, MPI_FLOAT, rank == 0 ? PostProcessGray.data() : nullptr,
              recvcounts.data(), displs_out.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

  /*if (rank == 0) {
    const auto &inp = GetInput();
    const auto &image = std::get<0>(inp);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        std::cout<< "{" << image[i][j].r<<","<<image[i][j].g<<","<<image[i][j].b<< "}";
      }
      std::cout << "\n";
    }
    std::cout << "\n";*/
  /*for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      std::cout << PostProcessGray[i*m+j] << " ";
    }
    std::cout << "\n";
  }
}*/

  return true;
}

bool ChaschinVSobelOperatorMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto &out = GetOutput();
  int n = std::get<0>(Size);
  int m = std::get<1>(Size);
  if (rank != 0) {
    PostProcessGray.resize(n * m);
  }
  MPI_Bcast(PostProcessGray.data(),  // буфер
            n * m,                   // количество элементов
            MPI_FLOAT,
            0,  // root процесс
            MPI_COMM_WORLD);

  out.resize(n);

  for (int i = 0; i < n; ++i) {
    out[i].resize(m);
    for (int j = 0; j < m; ++j) {
      float v = PostProcessGray[i * m + j];
      unsigned char c = static_cast<unsigned char>(std::clamp(v, 0.0f, 255.0f));
      out[i][j] = Pixel{c, c, c};
    }
  }

  return true;
}

}  // namespace chaschin_v_sobel_operator
