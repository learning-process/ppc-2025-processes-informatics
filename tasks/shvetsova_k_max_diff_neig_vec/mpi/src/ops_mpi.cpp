#include "shvetsova_k_max_diff_neig_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
// #include <cmath>

#include "shvetsova_k_max_diff_neig_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shvetsova_k_max_diff_neig_vec {

ShvetsovaKMaxDiffNeigVecMPI::ShvetsovaKMaxDiffNeigVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::pair<double, double>{0.0, 0.0};
}

bool ShvetsovaKMaxDiffNeigVecMPI::ValidationImpl() {
  return true;
}

bool ShvetsovaKMaxDiffNeigVecMPI::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

bool ShvetsovaKMaxDiffNeigVecMPI::RunImpl() {
  int count_of_proc;
  int rank;
  int size_of_vector = data_.size();
  if (size_of_vector < 2) {
    GetOutput().first = 0.0;
    GetOutput().second = 0.0;
    return true;
  }
  MPI_Comm_size(MPI_COMM_WORLD, &count_of_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Bcast(&size_of_vector, 1, MPI_INT, 0, MPI_COMM_WORLD);  // сказали всем процессам, какой размер у вектора
  std::vector<int> count_elems(count_of_proc, 0);             // количсетво элементов на каждый из процессов
  std::vector<int> ind(count_of_proc, 0);  // индекс, начиная с которого элементы принадлежат процессам

  if (rank == 0) {
    int base = size_of_vector / count_of_proc;
    int rem = size_of_vector % count_of_proc;
    int sum = 0;

    for (int i = 0; i < count_of_proc; i++) {
      count_elems[i] = base + (i < rem ? 1 : 0);
      ind[i] = sum;
      sum += count_elems[i];
    }
  }

  MPI_Bcast(count_elems.data(), count_of_proc, MPI_INT, 0,
            MPI_COMM_WORLD);  // разослали процессам информацию о кол-ве элементов
  MPI_Bcast(ind.data(), count_of_proc, MPI_INT, 0,
            MPI_COMM_WORLD);  // разослали порцессам индексы, с которых начинаются их элементы

  int part_size = count_elems[rank];    // определяяем кол-во элементов для текущего процесса
  std::vector<double> part(part_size);  // создаем вектор с размером = кол-во элементов для процесса

  MPI_Scatterv(data_.data(), count_elems.data(), ind.data(), MPI_DOUBLE, part.data(), part_size, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);  // распилили вектор

  // ищем локальные результаты
  double local_diff = -1.0;
  double local_a = 0.0, local_b = 0.0;

  for (int i = 0; i < part_size - 1; i++) {
    double diff = std::abs(part[i] - part[i + 1]);
    if (diff > local_diff) {
      local_diff = diff;
      local_a = part[i];
      local_b = part[i + 1];
    }
  }

  // обрабатиываем то, что на границах кусочков в процессах
  if (count_of_proc > 1) {
    if (rank < count_of_proc - 1 && part_size > 0) {
      double last_curr = part[part_size - 1];
      MPI_Send(&last_curr, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
    }

    if (rank > 0) {
      double last_prev = 0;
      MPI_Recv(&last_prev, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if (part_size > 0) {
        double diff = std::abs(last_prev - part[0]);
        if (diff > local_diff) {
          local_diff = diff;
          local_a = last_prev;
          local_b = part[0];
        }
      }
    }
  }

  // собираем локальные рузультаты
  std::vector<double> all_diffs(count_of_proc);
  MPI_Gather(&local_diff, 1, MPI_DOUBLE, all_diffs.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // на 0 процессе вычисляем глобальный результат
  double global_diff = 0.0;
  int winner_rank = 0;

  if (rank == 0) {
    global_diff = *std::max_element(all_diffs.begin(), all_diffs.end());

    for (int i = 0; i < count_of_proc; i++) {
      if (std::abs(all_diffs[i] - global_diff) < 1e-12) {
        winner_rank = i;
        break;
      }
    }
  }

  MPI_Bcast(&winner_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);  // рассылаем номер процесса победителя

  std::vector<double> all_pairs(count_of_proc * 2);
  double local_pair[2] = {local_a, local_b};

  MPI_Gather(local_pair, 2, MPI_DOUBLE, all_pairs.data(), 2, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);  // получаем все локальные пары результаты (где максимальная разница на прооцессе)

  double result_pair[2] = {0.0, 0.0};

  if (rank == 0) {
    // вычисляем глобальную пару с максимальной разницей
    result_pair[0] = all_pairs[winner_rank * 2];
    result_pair[1] = all_pairs[winner_rank * 2 + 1];
  }

  // рассылаем результат всем процессам
  MPI_Bcast(result_pair, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = {result_pair[0], result_pair[1]};

  return true;
}

bool ShvetsovaKMaxDiffNeigVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_max_diff_neig_vec
