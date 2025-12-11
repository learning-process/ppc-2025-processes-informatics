#include "nikitina_v_quick_sort_merge/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

namespace nikitina_v_quick_sort_merge {

TestTaskSEQ::TestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TestTaskSEQ::ValidationImpl() {
  return true;
}

bool TestTaskSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

bool TestTaskSEQ::RunImpl() {
  if (GetOutput().empty()) {
    return true;
  }
  std::sort(GetOutput().begin(), GetOutput().end());
  return true;
}

bool TestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace nikitina_v_quick_sort_merge
