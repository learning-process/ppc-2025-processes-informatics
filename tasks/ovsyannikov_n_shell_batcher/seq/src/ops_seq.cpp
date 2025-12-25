#include "ovsyannikov_n_shell_batcher/seq/include/ops_seq.hpp"
#include <algorithm>
#include <vector>

namespace ovsyannikov_n_shell_batcher {

// Конструктор: просто принимаем данные
OvsyannikovNShellBatcherSEQ::OvsyannikovNShellBatcherSEQ(const std::vector<int>& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool OvsyannikovNShellBatcherSEQ::ValidationImpl() {
  return true;
}

bool OvsyannikovNShellBatcherSEQ::PreProcessingImpl() {
  // НИКАКИХ MPI_Bcast, MPI_Comm_rank и т.д.
  // Просто готовим выходной вектор
  GetOutput() = GetInput();
  return true;
}

bool OvsyannikovNShellBatcherSEQ::RunImpl() {
  auto &arr = GetOutput();
  int n = static_cast<int>(arr.size());
  if (n < 2) return true;

  // Обычная последовательная сортировка Шелла.
  // В режиме pipeline этот код будет запущен фреймворком на одном ядре (Rank 0).
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