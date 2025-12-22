#include "maslova_u_fast_sort_simple/seq/include/ops_seq.hpp"
#include <algorithm>
#include <vector>

namespace maslova_u_fast_sort_simple {

MaslovaUFastSortSimpleSEQ::MaslovaUFastSortSimpleSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MaslovaUFastSortSimpleSEQ::ValidationImpl() {
  // Просто проверяем входные данные на Rank 0 (единственном, где выполняется SEQ)
  return true; 
}

bool MaslovaUFastSortSimpleSEQ::PreProcessingImpl() {
  // Инициализируем выходной вектор входными данными
  GetOutput() = GetInput();
  return true;
}

bool MaslovaUFastSortSimpleSEQ::RunImpl() {
  // Чистая сортировка без MPI
  if (!GetOutput().empty()) {
    std::sort(GetOutput().begin(), GetOutput().end());
  }
  return true;
}

bool MaslovaUFastSortSimpleSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace maslova_u_fast_sort_simple