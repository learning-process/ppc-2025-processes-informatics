#include "sannikov_i_column_sum/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "sannikov_i_column_sum/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sannikov_i_column_sum {

SannikovIColumnSumSEQ::SannikovIColumnSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SannikovIColumnSumSEQ::ValidationImpl() {
  return (!GetInput().empty()) && (GetInput().front().size() != 0) && (GetOutput().empty());
}

bool SannikovIColumnSumSEQ::PreProcessingImpl() {
  GetOutput().resize(GetInput().front().size(), 0);
  return !GetOutput().empty();
}

bool SannikovIColumnSumSEQ::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  for (size_t i = 0; i < (size_t)GetInput().size(); i++) {
    for (size_t j = 0; j < (size_t)GetInput()[i].size(); j++) {
      GetOutput()[j] += GetInput()[i][j];
    }
  }

  return !GetOutput().empty();
}

bool SannikovIColumnSumSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace sannikov_i_column_sum
