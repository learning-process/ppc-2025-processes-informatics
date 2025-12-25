#include "ovsyannikov_n_shell_batcher/seq/include/ops_seq.hpp"

#include <algorithm>

namespace ovsyannikov_n_shell_batcher {

OvsyannikovNShellBatcherSEQ::OvsyannikovNShellBatcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool OvsyannikovNShellBatcherSEQ::ValidationImpl() {
  // ЗДЕСЬ НЕ ДОЛЖНО БЫТЬ MPI_Bcast или MPI_Barrier
  return true;
}

bool OvsyannikovNShellBatcherSEQ::PreProcessingImpl() {
  // Просто копируем данные. Фреймворк PPC уже положил их в GetInput() на Rank 0.
  GetOutput() = GetInput();
  return true;
}

bool OvsyannikovNShellBatcherSEQ::RunImpl() {
  auto &arr = GetOutput();
  int n = static_cast<int>(arr.size());
  if (n < 2) {
    return true;
  }

  // Обычная сортировка Шелла без MPI
  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; i++) {
      int temp = arr[i];
      int j = i;
      while (j >= gap && arr[j - gap] > temp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = temp;
    }
  }
  return true;
}

bool OvsyannikovNShellBatcherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace ovsyannikov_n_shell_batcher
