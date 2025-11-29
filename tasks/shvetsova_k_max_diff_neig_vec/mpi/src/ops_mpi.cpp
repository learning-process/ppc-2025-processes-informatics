#include "shvetsova_k_max_diff_neig_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "shvetsova_k_max_diff_neig_vec/common/include/common.hpp"

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
  int count_of_proc = 0;
  int rank = 0;
  // int size_of_vector = static_cast<int>(data_.size());

  MPI_Comm_size(MPI_COMM_WORLD, &count_of_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size_of_vector = 0;
  if (rank == 0) {
    size_of_vector = static_cast<int>(data_.size());
  }

  MPI_Bcast(&size_of_vector, 1, MPI_INT, 0, MPI_COMM_WORLD);  // сказали всем процессам, какой размер у вектора
  if (size_of_vector < 2) {
    if (rank == 0) {
      GetOutput().first = 0.0;
      GetOutput().second = 0.0;
    }
    return true;
  }
  std::vector<int> count_elems(count_of_proc, 0);  // количсетво элементов на каждый из процессов
  std::vector<int> ind(count_of_proc, 0);          // индекс, начиная с которого элементы принадлежат процессам

  CreateDistribution(count_of_proc, size_of_vector, count_elems, ind, rank);

  int part_size = count_elems[rank];
  std::vector<double> part(part_size);

  MPI_Scatterv(rank == 0 ? data_.data() : nullptr, count_elems.data(), ind.data(), MPI_DOUBLE,
               part_size > 0 ? part.data() : nullptr, part_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double local_diff = -1.0;
  double local_a = 0.0;
  double local_b = 0.0;

  ComputeLocalDiff(part, part_size, local_diff, local_a, local_b);
  ComputeBorders(count_of_proc, rank, part, part_size, local_diff, local_a, local_b);

  std::vector<double> all_diffs(count_of_proc);
  MPI_Gather(&local_diff, 1, MPI_DOUBLE, rank == 0 ? all_diffs.data() : nullptr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int winner_rank = WinnerRank(all_diffs, count_of_proc, rank);

  std::array<double, 2> result_pair{};
  CollectGlobalPair(winner_rank, local_a, local_b, result_pair, count_of_proc);

  MPI_Bcast(result_pair.data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = {result_pair[0], result_pair[1]};
  return true;
}

// Реализация функций
void ShvetsovaKMaxDiffNeigVecMPI::CreateDistribution(int count_of_proc, int size_of_vector,
                                                     std::vector<int> &count_elems, std::vector<int> &ind, int rank) {
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

  MPI_Bcast(count_elems.data(), count_of_proc, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(ind.data(), count_of_proc, MPI_INT, 0, MPI_COMM_WORLD);
}

void ShvetsovaKMaxDiffNeigVecMPI::ComputeBorders(int count_of_proc, int rank, const std::vector<double> &part,
                                                 int part_size, double &local_diff, double &local_a, double &local_b) {
  if (count_of_proc == 1) {
    ComputeBordersSingleProcess(part, part_size, local_diff, local_a, local_b);
    return;
  }

  // -------- 1. Готовим данные для отправки --------
  double send_val = 0.0;
  int send_count = 0;

  if (part_size > 0) {
    send_val = part[part_size - 1];  // последний элемент
    send_count = 1;
  }

  // -------- 2. Определяем соседей --------
  int dest = (rank < count_of_proc - 1) ? rank + 1 : MPI_PROC_NULL;
  int src = (rank > 0) ? rank - 1 : MPI_PROC_NULL;

  // -------- 3. Обмениваемся безопасно --------
  double recv_val = 0.0;

  MPI_Sendrecv(&send_val, send_count, MPI_DOUBLE, dest, 0,  // отправка
               &recv_val, 1, MPI_DOUBLE, src, 0,            // приём
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // -------- 4. Если слева реально пришли данные — считаем разницу --------
  if (src != MPI_PROC_NULL && part_size > 0) {
    double diff = std::abs(recv_val - part[0]);

    if (diff > local_diff) {
      local_diff = diff;
      local_a = recv_val;
      local_b = part[0];
    }
  }
}

void ShvetsovaKMaxDiffNeigVecMPI::ComputeBordersSingleProcess(const std::vector<double> &part, int part_size,
                                                              double &local_diff, double &local_a, double &local_b) {
  if (part_size < 2) {
    return;
  }

  for (int i = 0; i + 1 < part_size; ++i) {
    double diff = std::abs(part[i] - part[i + 1]);
    if (diff > local_diff) {
      local_diff = diff;
      local_a = part[i];
      local_b = part[i + 1];
    }
  }
}

void ShvetsovaKMaxDiffNeigVecMPI::ComputeLocalDiff(const std::vector<double> &part, int part_size, double &local_diff,
                                                   double &local_a, double &local_b) {
  for (int i = 0; i < part_size - 1; i++) {
    double diff = std::abs(part[i] - part[i + 1]);
    if (diff > local_diff) {
      local_diff = diff;
      local_a = part[i];
      local_b = part[i + 1];
    }
  }
}

int ShvetsovaKMaxDiffNeigVecMPI::WinnerRank(const std::vector<double> &all_diffs, int count_of_proc, int rank) {
  int winner_rank = 0;

  if (rank == 0) {
    double global_diff = *std::ranges::max_element(all_diffs);
    for (int i = 0; i < count_of_proc; i++) {
      if (std::abs(all_diffs[i] - global_diff) < 1e-12) {
        winner_rank = i;
        break;
      }
    }
  } else {
    winner_rank = 0;
  }

  MPI_Bcast(&winner_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return winner_rank;
}

void ShvetsovaKMaxDiffNeigVecMPI::CollectGlobalPair(int winner_rank, double local_a, double local_b,
                                                    std::array<double, 2> &result_pair, int count_of_proc) {
  std::array<double, 2> local_pair = {local_a, local_b};
  std::vector<double> all_pairs(static_cast<size_t>(count_of_proc) * 2);

  MPI_Gather(local_pair.data(), 2, MPI_DOUBLE, all_pairs.data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(&winner_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

  result_pair[0] = all_pairs[static_cast<size_t>(winner_rank) * 2];
  result_pair[1] = all_pairs[(static_cast<size_t>(winner_rank) * 2) + 1];
}

bool ShvetsovaKMaxDiffNeigVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_max_diff_neig_vec
