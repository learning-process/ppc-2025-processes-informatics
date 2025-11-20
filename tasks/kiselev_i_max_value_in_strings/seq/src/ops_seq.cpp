#include "kiselev_i_max_value_in_strings/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "kiselev_i_max_value_in_strings/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kiselev_i_max_value_in_strings {

KiselevITestTaskSEQ::KiselevITestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput();
}

bool KiselevITestTaskSEQ::ValidationImpl() {
  const auto &matrix =GetInput();
  if(matrix.empty()){
    return false;
  }
  return true;
}

bool KiselevITestTaskSEQ::PreProcessingImpl() {
  const auto &matrix =GetInput();
  GetOutput().resize(matrix.size());
  return true;
}

bool KiselevITestTaskSEQ::RunImpl() {
  const auto &matrix=GetInput();
  auto &out_vector =GetOutput();
  for(int num_str = 0; num_str < matrix.size(); num_str++){
    int tmp_max_elem_in_str= matrix[num_str][0];
    for( int num_column = 0; num_column < matrix[num_str].size(); num_column++){
      if(matrix[num_str][num_column] > tmp_max_elem_in_str){
        tmp_max_elem_in_str=matrix[num_str][num_column];
      }
    }
    out_vector[num_str]=tmp_max_elem_in_str;
  }
  return true;
}

bool KiselevITestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kiselev_i_max_value_in_strings
