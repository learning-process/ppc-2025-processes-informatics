#include "votincev_d_qsort_batcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace votincev_d_qsort_batcher {

VotincevDQsortBatcherMPI::VotincevDQsortBatcherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

// убеждаемся, что массив не пустой
bool VotincevDQsortBatcherMPI::ValidationImpl() {
  const auto &vec = GetInput();
  return !vec.empty();
}

// препроцессинг (не нужен)
bool VotincevDQsortBatcherMPI::PreProcessingImpl() {
  return true;
}

// главный MPI метод
bool VotincevDQsortBatcherMPI::RunImpl() {
  // получаю кол-во процессов
  int proc_n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_n);

  // получаю ранг процесса
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // размер массива знает только 0-й процесс
  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(GetInput().size());
  }

  // рассылаем размер массива всем процессам
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    return true;
  }

  // если процесс 1 - то просто как в SEQ
  if (proc_n == 1) {
    if (rank == 0) {
      auto out = GetInput();
      QuickSort(out.data(), 0, static_cast<int>(out.size()) - 1);
      GetOutput() = out;
    }
    return true;
  }

  // вычисление размеров и смещений
  std::vector<int> sizes;
  std::vector<int> offsets;
  ComputeDistribution(proc_n, total_size, sizes, offsets);

  // распределение данных (Scatter)
  std::vector<double> local(sizes[rank]);
  ScatterData(rank, sizes, offsets, local);

  // локальная сортировка
  // каждый процесс сортирует свой кусок
  if (!local.empty()) {
    QuickSort(local.data(), 0, static_cast<int>(local.size()) - 1);
  }

  // чет-нечет слияние Бэтчера
  BatcherMergeSort(rank, proc_n, sizes, local);

  // 0й процесс собирает результыт от других процессов
  GatherResult(rank, total_size, sizes, offsets, local);

  return true;
}

void VotincevDQsortBatcherMPI::ComputeDistribution(int proc_n, int total_size, std::vector<int> &sizes,
                                                   std::vector<int> &offsets) {
  int base = total_size / proc_n;   // минимальный размер блока
  int extra = total_size % proc_n;  // оставшиеся элементы

  sizes.resize(proc_n);
  offsets.resize(proc_n);

  // распределяем остаток по первым процессам (процессам с меньшим рангом)
  for (int i = 0; i < proc_n; i++) {
    sizes[i] = base + (i < extra ? 1 : 0);
  }

  //  смещения (начальный индекс для каждого блока)
  offsets[0] = 0;
  for (int i = 1; i < proc_n; i++) {
    offsets[i] = offsets[i - 1] + sizes[i - 1];
  }
}

void VotincevDQsortBatcherMPI::ScatterData(int rank, const std::vector<int> &sizes, const std::vector<int> &offsets,
                                           std::vector<double> &local) {
  //  GetInput().data() только для процесса с rank == 0
  if (rank == 0) {
    MPI_Scatterv(GetInput().data(), sizes.data(), offsets.data(), MPI_DOUBLE, local.data(), sizes[rank], MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, sizes.data(), offsets.data(), MPI_DOUBLE, local.data(), sizes[rank], MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
  }
}

void VotincevDQsortBatcherMPI::BatcherMergeSort(int rank, int proc_n, const std::vector<int> &sizes,
                                                std::vector<double> &local) {
  if (proc_n <= 1) {  // нечего сортировать
    return;
  }

  // массивы для обмена и слияния. Их размер определяется максимальным блоком.
  int max_block = *std::ranges::max_element(sizes);
  std::vector<double> recv_buf(max_block);
  std::vector<double> merge_buf(sizes[rank] + max_block);

  // чет-нечет слияние Бэтчера (P фаз)
  for (int phase = 0; phase < proc_n; phase++) {
    int partner = -1;

    // опрределяем соседа для текущей фазы
    if (phase % 2 == 0) {
      // чётная фаза: 0-1, 2-3, ...
      partner = (rank % 2 == 0) ? rank + 1 : rank - 1;
    } else {
      // нечётная фаза: 1-2, 3-4, ...
      partner = (rank % 2 == 1) ? rank + 1 : rank - 1;
    }

    // если соседа нет или он вне границ — пропускаем фазу
    if (partner < 0 || partner >= proc_n) {
      continue;
    }

    // обмениваемся отсортированными блоками с соседом
    MPI_Sendrecv(local.data(), sizes[rank], MPI_DOUBLE, partner, 0, recv_buf.data(), sizes[partner], MPI_DOUBLE,
                 partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // сливаем два отсортированных массива в merge_buf
    std::merge(local.begin(), local.end(), recv_buf.begin(), recv_buf.begin() + sizes[partner], merge_buf.begin());

    // распределение объединенного блока:
    if (rank < partner) {
      // младший ранг (меньший индекс) оставляет меньшую часть (начало merge_buf)
      std::copy(merge_buf.begin(), merge_buf.begin() + sizes[rank], local.begin());
    } else {
      // старший ранг (больший индекс) оставляет большую часть (конец merge_buf)
      std::copy(merge_buf.begin() + sizes[partner], merge_buf.begin() + sizes[partner] + sizes[rank], local.begin());
    }
  }
}

// 0й процесс собирает данные со всех других процессов
void VotincevDQsortBatcherMPI::GatherResult(int rank, int total_size, const std::vector<int> &sizes,
                                            const std::vector<int> &offsets, const std::vector<double> &local) {
  if (rank == 0) {
    std::vector<double> result(total_size);

    // 0-й процесс собирает данные
    MPI_Gatherv(local.data(), sizes[rank], MPI_DOUBLE, result.data(), sizes.data(), offsets.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    GetOutput() = result;
  } else {
    // остальные процессы отправляют свои данные 0-му
    MPI_Gatherv(local.data(), sizes[rank], MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

// итеративная qsort
void VotincevDQsortBatcherMPI::QuickSort(double *arr, int left, int right) {
  std::vector<int> stack;

  stack.push_back(left);
  stack.push_back(right);

  while (!stack.empty()) {
    int h = stack.back();
    stack.pop_back();
    int l = stack.back();
    stack.pop_back();

    if (l >= h) {
      continue;
    }

    int i = l;
    int j = h;
    double pivot = arr[(l + h) / 2];

    while (i <= j) {
      while (arr[i] < pivot) {
        i++;
      }
      while (arr[j] > pivot) {
        j--;
      }

      if (i <= j) {
        std::swap(arr[i], arr[j]);
        i++;
        j--;
      }
    }

    if (l < j) {
      stack.push_back(l);
      stack.push_back(j);
    }

    if (i < h) {
      stack.push_back(i);
      stack.push_back(h);
    }
  }
}

// здесь ничего не надо делать
bool VotincevDQsortBatcherMPI::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_qsort_batcher
