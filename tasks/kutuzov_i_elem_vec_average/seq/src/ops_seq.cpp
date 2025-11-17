#include "kutuzov_i_elem_vec_average/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "kutuzov_i_elem_vec_average/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kutuzov_i_elem_vec_average {

KutuzovIElemVecAverageSEQ::KutuzovIElemVecAverageSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool KutuzovIElemVecAverageSEQ::ValidationImpl() {
  return GetInput().size() > 0;
}

bool KutuzovIElemVecAverageSEQ::PreProcessingImpl() {
  return true;
}

bool KutuzovIElemVecAverageSEQ::RunImpl() {
  GetOutput() = 0.0;
  for (size_t i = 0; i < GetInput().size(); i++) {
    GetOutput() += GetInput()[i];
  }

  GetOutput() /= GetInput().size();

  return true;
}

bool KutuzovIElemVecAverageSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kutuzov_i_elem_vec_average
