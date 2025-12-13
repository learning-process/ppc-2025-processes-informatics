#include "votincev_d_qsort_batcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>  // max_element, merge, copy
#include <numeric>
#include <vector>

namespace votincev_d_qsort_batcher {

// ручная быстрая сортировка
// обычный quicksort для массива double
// сортирует элементы на месте
void VotincevDQsortBatcherMPI::QuickSort(double *arr, int left, int right) {
  int i = left;
  int j = right;

  // берем опорный элемент из середины
  double pivot = arr[(left + right) / 2];

  // разделяем массив на две части
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

  // рекурсивно сортируем левую и правую части
  if (left < j) {
    QuickSort(arr, left, j);
  }
  if (i < right) {
    QuickSort(arr, i, right);
  }
}

VotincevDQsortBatcherMPI::VotincevDQsortBatcherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

// проверка входных данных
// просто убеждаемся, что массив не пустой
bool VotincevDQsortBatcherMPI::ValidationImpl() {
  const auto &vec = GetInput();
  return !vec.empty();
}

// препроцессинг
bool VotincevDQsortBatcherMPI::PreProcessingImpl() {
  return true;
}

// главный MPI метод
bool VotincevDQsortBatcherMPI::RunImpl() {
  // получаем общее число процессов
  int proc_n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_n);

  // получаем ранг текущего процесса
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // размер массива знает только 0й процесс
  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(GetInput().size());
  }

  // рассылаем размер массива всем процессам
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // ---------- случай одного процесса ----------
  // просто обычная сортировка без MPI
  if (proc_n == 1) {
    if (rank == 0) {
      auto out = GetInput();
      QuickSort(out.data(), 0, static_cast<int>(out.size()) - 1);
      GetOutput() = out;
    }
    return true;
  }

  // ---------- вычисляем размеры блоков ----------
  std::vector<int> sizes(proc_n);
  std::vector<int> offsets(proc_n);

  int base = total_size / proc_n;   // минимальный размер блока
  int extra = total_size % proc_n;  // остаток

  // распределяем остаток по первым процессам
  for (int i = 0; i < proc_n; i++) {
    sizes[i] = base + (i < extra ? 1 : 0);
  }

  // считаем смещения
  offsets[0] = 0;
  for (int i = 1; i < proc_n; i++) {
    offsets[i] = offsets[i - 1] + sizes[i - 1];
  }

  // ---------- Scatter ----------
  // каждый процесс получает свой кусок массива
  std::vector<double> local(sizes[rank]);

  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, sizes.data(), offsets.data(), MPI_DOUBLE, local.data(),
               sizes[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // ---------- локальная сортировка ----------
  // сортируем свой кусок вручную
  if (!local.empty()) {
    QuickSort(local.data(), 0, static_cast<int>(local.size()) - 1);
  }

  // ---------- буферы ----------
  // максимальный размер блока
  int max_block = *std::max_element(sizes.begin(), sizes.end());

  // буфер для приема данных от соседа
  std::vector<double> recv_buf(max_block);

  // буфер для слияния двух отсортированных блоков
  std::vector<double> merge_buf(sizes[rank] + max_block);

  // ---------- четно-нечетное слияние Бэтчера ----------
  for (int phase = 0; phase < proc_n; phase++) {
    int partner = -1;

    // определяем соседа для текущей фазы
    if (phase % 2 == 0) {
      partner = (rank % 2 == 0) ? rank + 1 : rank - 1;
    } else {
      partner = (rank % 2 == 1) ? rank + 1 : rank - 1;
    }

    // если соседа нет — пропускаем фазу
    if (partner < 0 || partner >= proc_n) {
      continue;
    }

    // обмениваемся отсортированными блоками
    MPI_Sendrecv(local.data(), sizes[rank], MPI_DOUBLE, partner, 0, recv_buf.data(), sizes[partner], MPI_DOUBLE,
                 partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // сливаем два отсортированных массива
    std::merge(local.begin(), local.end(), recv_buf.begin(), recv_buf.begin() + sizes[partner], merge_buf.begin());

    // младший ранг оставляет меньшую часть
    if (rank < partner) {
      std::copy(merge_buf.begin(), merge_buf.begin() + sizes[rank], local.begin());
    }
    // старший ранг оставляет большую часть
    else {
      std::copy(merge_buf.begin() + sizes[partner], merge_buf.begin() + sizes[partner] + sizes[rank], local.begin());
    }
  }

  // ---------- Gather ----------
  // собираем отсортированные блоки обратно
  if (rank == 0) {
    std::vector<double> result(total_size);

    MPI_Gatherv(local.data(), sizes[rank], MPI_DOUBLE, result.data(), sizes.data(), offsets.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    GetOutput() = result;
  } else {
    // остальные процессы просто отправляют свои данные
    MPI_Gatherv(local.data(), sizes[rank], MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool VotincevDQsortBatcherMPI::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_qsort_batcher
