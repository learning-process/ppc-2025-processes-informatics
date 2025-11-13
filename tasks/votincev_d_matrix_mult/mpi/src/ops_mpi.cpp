#include "votincev_d_matrix_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <algorithm>
#include <tuple>
#include <vector>

#include "votincev_d_matrix_mult/common/include/common.hpp"

namespace votincev_d_matrix_mult {

VotincevDMatrixMultMPI::VotincevDMatrixMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

// проверка входных данных
bool VotincevDMatrixMultMPI::ValidationImpl() {
  const auto &in = GetInput();
  int m = std::get<0>(in);
  int n = std::get<1>(in);
  int k = std::get<2>(in);
  const auto &A = std::get<3>(in);
  const auto &B = std::get<4>(in);

  return (m > 0 && n > 0 && k > 0 &&
          A.size() == static_cast<size_t>(m * k) &&
          B.size() == static_cast<size_t>(k * n));
}

// препроцессинг
bool VotincevDMatrixMultMPI::PreProcessingImpl() {
  auto &in = GetInput();
  m_ = std::get<0>(in);
  n_ = std::get<1>(in);
  k_ = std::get<2>(in);
  A_ = std::get<3>(in);
  B_ = std::get<4>(in);

  result_.assign(m_ * n_, 0.0);
  return true;
}

// главный метод MPI
bool VotincevDMatrixMultMPI::RunImpl() {
  // получаю кол-во процессов
  int process_n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &process_n);

  // получаю ранг текущего
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  // если всего 1 процесс -  последовательное умножение
  if (process_n == 1) {
    MultiplyBlock(0, m_, result_);
    return true;
  }

  // какие строки каждый процесс будет перемножать
  // [start0, end0, start1, end1, ...]
  std::vector<int> ranges;
  if (proc_rank == 0) {
    ranges.resize(process_n * 2); // *2 так как каждый процесс получит start и end свой

    // минимум на обработку
    int base = m_ / process_n;
    // остаток - распределим между первыми
    int remain = m_ % process_n;

    int start = 0;
    for (int i = 0; i < process_n; i++) {
      // Количество строк для процесса i
      int part = base;
      if (i < remain) { // первые remain процессво получат по +1
        part++;
      }

      
      ranges[i * 2] = start; // строка start для процесса i
      ranges[i * 2 + 1] = start + part; // строка end для процесса i
      // процессы обрабатывают строки с start по end-1 !!! 


      start += part;
    }
  }

  // диапазон для процесса
  int my_range[2]{0, 0};
  MPI_Scatter(
      ranges.data(),  // что отправляем ( std::vector<int> ranges)
      2,              // сколько каждый получит (по 2 типа int)
      MPI_INT,        // тип отправляемых данных
      my_range,       // куда класть 
      2,              // сколько класть
      MPI_INT,        // тип что кладем
      0,              // тег
      MPI_COMM_WORLD); // коммуникатор


  // каждый процесс берет свой start и end
  int start = my_range[0];
  int end = my_range[1];
  int local_rows = end - start;  // сколько будет обрабатывать строк

  // результирующая матрица процесса (часть выходной матрицы)
  std::vector<double> local_R(local_rows * n_, 0.0);

  // произведение строк start ... end-1 матрцы А и матрицы B, пишу в local_R
  MultiplyBlock(start, end, local_R);



  // ???????????????????
  //
  эточё
  // Подготавливаем данные для сборки результата у процесса 0
  std::vector<int> recvcounts, displs;
  if (proc_rank == 0) {
    recvcounts.resize(process_n);  // сколько элементов приходит от каждого процесса
    displs.resize(process_n);      // смещения в итоговом буфере

    int offset = 0;
    for (int i = 0; i < process_n; i++) {
      int rs = ranges[i * 2 + 1] - ranges[i * 2];  // число строк от процесса i
      recvcounts[i] = rs * n_;                     // строк * столбцов = элементов
      displs[i] = offset;                          // где в result_ начинается его блок
      offset += recvcounts[i];
    }
  }

  // Сбор результата в итоговую матрицу result_ на процессе 0
  MPI_Gatherv(
      local_R.data(),    // отправляемые данные
      local_rows * n_,   // их количество
      MPI_DOUBLE,        // тип отправляемых
      result_.data(),    // куда собираем (только у 0-го)
      recvcounts.data(), // сколько от каждого процесса
      displs.data(),     // смещения
      MPI_DOUBLE,        // тип принимаемых
      0,                 // тег
      MPI_COMM_WORLD);   // коммуникатор

  // 0й процесс посылает остальным процессам результирующую матрицу
  SyncResults();

  return true;
}


// Синхронизация результата между процессами
void VotincevDMatrixMultMPI::SyncResults() {
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(result_.data(), m_ * n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}

// умножение части матрицы A на матрицу B
void VotincevDMatrixMultMPI::MultiplyBlock(int start_row, int end_row, std::vector<double> &out) {
  for (int i = start_row; i < end_row; ++i) {
    for (int j = 0; j < n_; ++j) {
      double sum = 0.0;
      for (int t = 0; t < k_; ++t) {
        sum += A_[i * k_ + t] * B_[t * n_ + j];
      }
      out[(i - start_row) * n_ + j] = sum;
    }
  }
}

bool VotincevDMatrixMultMPI::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace votincev_d_matrix_mult
