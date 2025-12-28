#include "ovsyannikov_n_shell_batcher/seq/include/ops_seq.hpp"

namespace ovsyannikov_n_shell_batcher {

OvsyannikovNShellBatcherSEQ::OvsyannikovNShellBatcherSEQ(const InType &in) {
  this->SetTypeOfTask(GetStaticTypeOfTask());
  this->GetInput() = in;
}

bool OvsyannikovNShellBatcherSEQ::ValidationImpl() {
  return true;
}

bool OvsyannikovNShellBatcherSEQ::PreProcessingImpl() {
  // Копируем входные данные в выходные
  this->GetOutput() = this->GetInput();
  return true;
}

void OvsyannikovNShellBatcherSEQ::ShellSort(std::vector<int> &arr) {
  int n = static_cast<int>(arr.size());
  if (n < 2) {
    return;
  }

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
}

bool OvsyannikovNShellBatcherSEQ::RunImpl() {
  auto &arr = this->GetOutput();
  ShellSort(arr);
  return true;
}

bool OvsyannikovNShellBatcherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace ovsyannikov_n_shell_batcher
