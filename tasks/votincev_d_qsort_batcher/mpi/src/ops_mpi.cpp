#include "votincev_d_qsort_batcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>
#include <cstddef>

namespace votincev_d_qsort_batcher {

VotincevDQsortBatcherMPI::VotincevDQsortBatcherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

// входные данные — просто std::vector<double>
bool VotincevDQsortBatcherMPI::ValidationImpl() {
  const auto &vec = GetInput();
  return !vec.empty();  // просто проверяем, что не пустой
}

bool VotincevDQsortBatcherMPI::PreProcessingImpl() {
  return true;
}

bool VotincevDQsortBatcherMPI::RunImpl() {
  int proc_n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_n);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const std::vector<double> &input = GetInput();
  int total_size = static_cast<int>(input.size());

  if (proc_n == 1) {
    std::vector<double> seq_sorted = input;
    std::sort(seq_sorted.begin(), seq_sorted.end());
    GetOutput() = seq_sorted;
    return true;
  }

  // ---------- деление массива ----------
  int base = total_size / proc_n;
  int extra = total_size % proc_n;

  std::vector<int> sizes(proc_n);
  std::vector<int> offsets(proc_n);

  for (int i = 0; i < proc_n; i++) {
    sizes[i] = base + (i < extra ? 1 : 0);
  }

  offsets[0] = 0;
  for (int i = 1; i < proc_n; i++) {
    offsets[i] = offsets[i - 1] + sizes[i - 1];
  }

  // ---------- получение локального блока ----------
  std::vector<double> local(sizes[rank]);
  if (rank == 0) {
    // 0-й шлёт всем
    for (int p = 1; p < proc_n; p++) {
      MPI_Send(input.data() + offsets[p], sizes[p], MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
    }
    std::copy(input.begin(), input.begin() + sizes[0], local.begin());
  } else {
    MPI_Recv(local.data(), sizes[rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // ---------- локальная сортировка ----------
  std::sort(local.begin(), local.end());

  // ---------- вспомогательный буфер ----------
  std::vector<double> recv_buf;
  recv_buf.reserve(base + 1);

  // ---------- odd-even Batcher merge ----------
  for (int phase = 0; phase < proc_n; phase++) {
    int partner = -1;

    if (phase % 2 == 0) {
      // чётная фаза
      if (rank % 2 == 0) partner = rank + 1;
      else partner = rank - 1;
    } else {
      // нечётная фаза
      if (rank % 2 == 1) partner = rank + 1;
      else partner = rank - 1;
    }

    if (partner < 0 || partner >= proc_n) {
      continue;  // нет соседа
    }

    recv_buf.resize(sizes[partner]);

    // обмен локальных отсортированных частей
    MPI_Sendrecv(local.data(), sizes[rank], MPI_DOUBLE, partner, 0,
                 recv_buf.data(), sizes[partner], MPI_DOUBLE, partner, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // слить два блока
    std::vector<double> merged;
    merged.resize(sizes[rank] + sizes[partner]);

    std::merge(local.begin(), local.end(),
               recv_buf.begin(), recv_buf.end(),
               merged.begin());

    // младший ранг — нижняя часть, старший — верхняя часть
    if (rank < partner) {
      // берём первые sizes[rank]
      local.assign(merged.begin(), merged.begin() + sizes[rank]);
    } else {
      // берём последние sizes[rank]
      local.assign(merged.end() - sizes[rank], merged.end());
    }
  }

  // ---------- сбор результата ----------
  if (rank == 0) {
    std::vector<double> result(total_size);

    std::copy(local.begin(), local.end(), result.begin());

    int offset = sizes[0];
    for (int p = 1; p < proc_n; p++) {
      MPI_Recv(result.data() + offset, sizes[p], MPI_DOUBLE, p, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      offset += sizes[p];
    }

    GetOutput() = result;
  } else {
    MPI_Send(local.data(), sizes[rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool VotincevDQsortBatcherMPI::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_qsort_batcher
