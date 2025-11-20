#include "kruglova_a_max_diff_adjacent/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "kruglova_a_max_diff_adjacent/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kruglova_a_max_diff_adjacent {

KruglovaAMaxDiffAdjacentMPI::KruglovaAMaxDiffAdjacentMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0f;
}

bool KruglovaAMaxDiffAdjacentMPI::ValidationImpl() {
  return true;
}

bool KruglovaAMaxDiffAdjacentMPI::PreProcessingImpl() {
  return true;
}

float KruglovaAMaxDiffAdjacentMPI::LocalMaxDiff(std::vector<float> &local_vec) {
  float local_max = 0.0f;

  if (local_vec.size() >= 2) {
    for (size_t i = 1; i < local_vec.size(); ++i) {
      float diff = std::abs(local_vec[i] - local_vec[i - 1]);
      if (diff > local_max) {
        local_max = diff;
      }
    }
  }

  return local_max;  // возвращаем всегда
}

bool KruglovaAMaxDiffAdjacentMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &vec = GetInput();
  int n = 0;

  if (rank == 0) {
    n = static_cast<int>(vec.size());
  }

  // Broadcast size
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n < 2) {
    // слишком мало элементов для вычисления соседних разниц
    if (rank == 0) {
      GetOutput() = 0.0f;
    }
    return true;
  }

  // --- MPI_Scatterv Setup ---

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);
  int base = n / size;
  int rem = n % size;

  for (int p = 0; p < size; ++p) {
    // Базовое количество элементов для этого процесса
    int count = base + (p < rem ? 1 : 0);

    // Смещение: начальный индекс в исходном векторе (vec)
    displs[p] = p * base + std::min(p, rem);

    // Количество отправляемых элементов.
    // Добавляем 1 для граничного элемента, если это не последний процесс и не конец вектора.
    if (displs[p] + count < n && p != size - 1) {
      sendcounts[p] = count + 1;  // Блок + 1 граничный элемент
    } else {
      sendcounts[p] = count;  // Последний блок или блок, включающий конец vec
    }
  }

  // Определяем размер локального вектора на основе того, что нам отправят
  int local_count = sendcounts[rank];
  std::vector<float> local_vec(local_count);

  // Scatter vector to workers using MPI_Scatterv
  // Используем vec.data() только на ранге 0. На других рангах он игнорируется.
  MPI_Scatterv(rank == 0 ? vec.data() : nullptr, sendcounts.data(), displs.data(), MPI_FLOAT, local_vec.data(),
               local_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // --- Local Calculation and Reduction ---

  float local_max = 0.0f;
  local_max = LocalMaxDiff(local_vec);

  float global_max = 0.0f;
  // Собираем локальные максимумы и находим глобальный максимум на ранге 0
  MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

  // ----------------------------
  // Broadcast result to all ranks
  // ----------------------------
  // Рассылаем глобальный результат всем процессам
  MPI_Bcast(&global_max, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool KruglovaAMaxDiffAdjacentMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_max_diff_adjacent
