#include "maslova_u_fast_sort_simple/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

namespace maslova_u_fast_sort_simple {

MaslovaUFastSortSimpleSEQ::MaslovaUFastSortSimpleSEQ(const InType &in) {
  SetTypeOfTask(ppc::task::TypeOfTask::kMPI);
  GetInput() = in;
}

bool MaslovaUFastSortSimpleSEQ::ValidationImpl() {
  return true;
}

bool MaslovaUFastSortSimpleSEQ::PreProcessingImpl() {
  return true;
}

bool MaslovaUFastSortSimpleSEQ::RunImpl() {
  GetOutput() = GetInput();

  // Сортируем
  if (!GetOutput().empty()) {
    std::sort(GetOutput().begin(), GetOutput().end());
  }
  return true;
}

bool MaslovaUFastSortSimpleSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace maslova_u_fast_sort_simple
