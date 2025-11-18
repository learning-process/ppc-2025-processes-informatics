#include "luzan_e_matrix_rows_sum/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "luzan_e_matrix_rows_sum/common/include/common.hpp"
#include "util/include/util.hpp"

namespace luzan_e_matrix_rows_sum {

LuzanEMatrixRowsSumSEQ::LuzanEMatrixRowsSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool LuzanEMatrixRowsSumSEQ::ValidationImpl() {
  size_t height = std::get<1>(GetInput());
  size_t width = std::get<2>(GetInput());

  return std::get<0>(GetInput()).size() == (height * width) && height != 0 && width != 0;
}

bool LuzanEMatrixRowsSumSEQ::PreProcessingImpl() {
  size_t height = std::get<1>(GetInput());
  GetOutput().resize(height);
  for (size_t row = 0; row < height; row++) {
    GetOutput()[row] = 0;
  }
  return true;
}

bool LuzanEMatrixRowsSumSEQ::RunImpl() {
  size_t height = std::get<1>(GetInput());
  size_t width = std::get<2>(GetInput());
  const std::tuple_element_t<0, InType> &mat = std::get<0>(GetInput());

  for (size_t row = 0; row < height; row++) {
    for (size_t col = 0; col < width; col++) {
      GetOutput()[row] += mat[width * row + col];
    }
  }
  return true;
}

bool LuzanEMatrixRowsSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace luzan_e_matrix_rows_sum
