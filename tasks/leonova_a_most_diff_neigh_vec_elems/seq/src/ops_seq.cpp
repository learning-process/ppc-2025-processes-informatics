#include "leonova_a_most_diff_neigh_vec_elems/seq/include/ops_seq.hpp"

#include <cstdlib>
#include <numeric>
#include <vector>

#include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"
#include "util/include/util.hpp"

namespace leonova_a_most_diff_neigh_vec_elems {

LeonovaAMostDiffNeighVecElemsSEQ::LeonovaAMostDiffNeighVecElemsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::tuple<int, int>(0, 0);
}

bool LeonovaAMostDiffNeighVecElemsSEQ::ValidationImpl() {
  return !GetInput().empty();
}  // проверка что данные валидны

bool LeonovaAMostDiffNeighVecElemsSEQ::PreProcessingImpl() {
  return true;
}  // предобработка данных - мне не нужно

bool LeonovaAMostDiffNeighVecElemsSEQ::RunImpl() {
  if (GetInput().size() == 1)  // в векторе всего один элемент => я хочу чтобы вывелся сам элемент в паре с собой - это
                               // не ошибка - самая большая разница это 0
  {
    std::get<0>(GetOutput()) = GetInput()[0];
    std::get<1>(GetOutput()) = GetInput()[0];
    return true;
  }

  int max_diff = -1;  // фиктивная разница
  for (std::vector<int>::size_type i = 0; i < GetInput().size() - 1; i++) {
    std::tuple<int, int> curr_elems(GetInput()[i], GetInput()[i + 1]);  // текущая пара елементов
    int curr_diff = abs(std::get<0>(curr_elems) - std::get<1>(curr_elems));  // разница очевидно считается по модулю
    if (curr_diff > max_diff)  // берем первую пару с наибольшей разницей в значениях
    {
      max_diff = curr_diff;
      GetOutput() = curr_elems;
    }
  }

  return true;
}  // основной код

bool LeonovaAMostDiffNeighVecElemsSEQ::PostProcessingImpl() {
  return true;
}  // постобработка данных - не нужна

}  // namespace leonova_a_most_diff_neigh_vec_elems
