#include "smyshlaev_a_gauss_filt/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "smyshlaev_a_gauss_filt/common/include/common.hpp"
#include "util/include/util.hpp"

namespace smyshlaev_a_gauss_filt {

SmyshlaevAGaussFiltSEQ::SmyshlaevAGaussFiltSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SmyshlaevAGaussFiltSEQ::ValidationImpl() {
  return true;
}

bool SmyshlaevAGaussFiltSEQ::PreProcessingImpl() {
  return true;
}

bool SmyshlaevAGaussFiltSEQ::RunImpl() {
  return true;
}

bool SmyshlaevAGaussFiltSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_gauss_filt
