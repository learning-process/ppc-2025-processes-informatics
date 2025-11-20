#include "kichanova_k_count_letters_in_str/seq/include/ops_seq.hpp"

#include <cctype>
#include <string>

#include "kichanova_k_count_letters_in_str/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kichanova_k_count_letters_in_str {

KichanovaKCountLettersInStrSEQ::KichanovaKCountLettersInStrSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KichanovaKCountLettersInStrSEQ::ValidationImpl() {
  return !GetInput().empty() && (GetOutput() == 0);
}

bool KichanovaKCountLettersInStrSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KichanovaKCountLettersInStrSEQ::RunImpl() {
const std::string& input_str = GetInput();
  
  for (char c : input_str) {
    if (std::isalpha(static_cast<unsigned char>(c))) {
      GetOutput()++;
    }
  }
  
  return GetOutput() >= 0;
}

bool KichanovaKCountLettersInStrSEQ::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace kichanova_k_count_letters_in_str
