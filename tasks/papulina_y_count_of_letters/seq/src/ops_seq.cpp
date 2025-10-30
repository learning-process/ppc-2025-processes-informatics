#include "papulina_y_count_of_letters/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "papulina_y_count_of_letters/common/include/common.hpp"
#include "util/include/util.hpp"

namespace papulina_y_count_of_letters {

PapulinaYCountOfLettersSEQ::PapulinaYCountOfLettersSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}
int PapulinaYCountOfLettersSEQ::CountOfLetters() {
  int k = 0;
  for (size_t i = 0; i < GetInput().size(); i++) {
    char c = GetInput()[i];
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
      k++;
    }
  }
  return k;
}
bool PapulinaYCountOfLettersSEQ::ValidationImpl() {
  return true;
}

bool PapulinaYCountOfLettersSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return GetOutput() == 0;
}

bool PapulinaYCountOfLettersSEQ::RunImpl() {
  GetOutput() = CountOfLetters();
  return true;
}

bool PapulinaYCountOfLettersSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace papulina_y_count_of_letters
