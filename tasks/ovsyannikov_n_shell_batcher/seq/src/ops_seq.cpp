#include "ovsyannikov_n_shell_batcher/seq/include/ops_seq.hpp"

#include "ovsyannikov_n_shell_batcher/common/include/common.hpp"

namespace ovsyannikov_n_shell_batcher {

OvsyannikovNShellBatcherSEQ::OvsyannikovNShellBatcherSEQ(const InType &in) {
  this->SetTypeOfTask(GetStaticTypeOfTask());
  this->GetInput() = in;
}

bool OvsyannikovNShellBatcherSEQ::ValidationImpl() {
  return true;
}

bool OvsyannikovNShellBatcherSEQ::PreProcessingImpl() {
  this->GetOutput() = this->GetInput();
  return true;
}

bool OvsyannikovNShellBatcherSEQ::RunImpl() {
  auto &arr = this->GetOutput();
  int n = static_cast<int>(arr.size());
  if (n < 2) {
    return true;
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
  return true;
}

bool OvsyannikovNShellBatcherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace ovsyannikov_n_shell_batcher
