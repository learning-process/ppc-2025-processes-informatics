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

  return (m > 0 && n > 0 && k > 0 && A.size() == static_cast<size_t>(m * k) && B.size() == static_cast<size_t>(k * n));
}

// препроцессинг
bool VotincevDMatrixMultMPI::PreProcessingImpl() {
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

  // размерности матриц получают все процессы
  int m = 0, n = 0, k = 0;
  const auto &in = GetInput();
  m = std::get<0>(in);
  n = std::get<1>(in);
  k = std::get<2>(in);

  // если процессов больше чем строк в матрице A -
  // то активных процессов будет меньше (m)
  //  (потому что разедление по строкам)
  process_n = std::min(process_n, m);

  // "лишние" процессы не работают
  if (proc_rank >= process_n) {
    return true;
  }

  std::vector<double> matrix_A;
  std::vector<double> matrix_B;

  // матрицу B получают все процессы
  matrix_B = std::get<4>(in);

  // матрицу А получит полностью только 0й процесс
  if (proc_rank == 0) {
    matrix_A = std::get<3>(in);
  }

  // если всего 1 процесс - последовательное умножение
  if (process_n == 1) {
    GetOutput() = SeqMatrixMult(m, n, k, matrix_A, matrix_B);
    return true;
  }

  // какие строки каждый процесс будет перемножать
  // [start0, end0, start1, end1, ...]
  std::vector<int> ranges;

  if (proc_rank == 0) {
    ranges.resize(process_n * 2);

    // минимум на обработку
    int base = m / process_n;
    // остаток распределим
    int remain = m % process_n;

    int start = 0;
    for (int i = 0; i < process_n; i++) {
      int part = base;
      if (i < remain) {
        part++;
      }

      ranges[i * 2] = start;
      ranges[i * 2 + 1] = start + part;  // end (не включительно)

      start += part;
    }
  }

  // my_range получит [start, end]
  int my_range[2]{0, 0};

  // local_matrix — локальный блок матрицы A данного процесса
  std::vector<double> local_matrix;

  if (proc_rank == 0) {
    // заполняю данные для себя — свои строки матрицы А
    int my_start = ranges[0];
    int my_end = ranges[1];
    int my_rows = my_end - my_start;

    local_matrix.resize(my_rows * k);
    for (int i = 0; i < my_rows * k; i++) {
      local_matrix[i] = matrix_A[i];
    }

    // рассылаю остальным
    for (int i = 1; i < process_n; i++) {
      int start_i = ranges[2 * i];
      int end_i = ranges[2 * i + 1];

      MPI_Send(&start_i, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      MPI_Send(&end_i, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

      int elem_count = (end_i - start_i) * k;

      MPI_Send(matrix_A.data() + start_i * k, elem_count, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }

  } else {
    // получаю диапазон
    MPI_Recv(&my_range[0], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&my_range[1], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int start_i = my_range[0];
    int end_i = my_range[1];
    int elem_count = (end_i - start_i) * k;

    local_matrix.resize(elem_count);

    // получаю матрицу (часть)
    MPI_Recv(local_matrix.data(), elem_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // теперь каждый владеет своим куском (local_matrix)
  // вызываем обычное перемножение, результат умножение кладется в local_matrix
  MatrixPartMult(k, n, local_matrix, matrix_B);

  // сбор результатов назад к 0му
  if (proc_rank == 0) {
    std::vector<double> final_result(m * n);
    int my_rows = ranges[1] - ranges[0];  // сколько строк
    // копирую        откуда          до куда                 куда
    std::copy(local_matrix.begin(), local_matrix.end(), final_result.begin());

    // смещение; изначально равно количеству уже записанных значений
    int offset = my_rows * n;
    for (int i = 1; i < process_n; ++i) {
      int start_i = ranges[2 * i];
      int end_i = ranges[2 * i + 1];
      int rows = end_i - start_i;

      // сколько в данной пачке элементов
      int count = rows * n;

      MPI_Recv(final_result.data() + offset, count, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      offset += count;
    }
    // 0й процесс собрал всё от других процессов

    GetOutput() = final_result;

  } else {
    // другие процессы посылают свои результаты основному 0му процессу
    int rows = my_range[1] - my_range[0];
    MPI_Send(local_matrix.data(), rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  return true;
}

// ===============================
// ==== дополнительные функции ===
// ===============================

// простое последовательное умножение (если кол-во_процессов == 1)
std::vector<double> VotincevDMatrixMultMPI::SeqMatrixMult(int param_m, int param_n, int param_k,
                                                          std::vector<double> &matrix_A,
                                                          std::vector<double> &matrix_B) {
  std::vector<double> matrix_res;
  matrix_res.assign(param_m * param_n, 0.0);

  for (int i = 0; i < param_m; ++i) {
    for (int j = 0; j < param_n; ++j) {
      double sum = 0.0;
      for (int t = 0; t < param_k; ++t) {
        sum += matrix_A[i * param_k + t] * matrix_B[t * param_n + j];
      }
      matrix_res[i * param_n + j] = sum;
    }
  }
  return matrix_res;
}

// умножение части матрицы A на всю матрицу B
void VotincevDMatrixMultMPI::MatrixPartMult(int k, int n, std::vector<double> &local_matrix,
                                            const std::vector<double> &matrix_B) {
  size_t str_count = local_matrix.size() / k;

  std::vector<double> result;
  result.resize(str_count * n);

  for (size_t i = 0; i < str_count; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int t = 0; t < k; t++) {
        sum += local_matrix[i * k + t] * matrix_B[t * n + j];
      }
      result[i * n + j] = sum;
    }
  }
  local_matrix = result;
}

bool VotincevDMatrixMultMPI::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_matrix_mult
