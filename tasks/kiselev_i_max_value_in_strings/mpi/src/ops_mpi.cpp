#include "kiselev_i_max_value_in_strings/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "kiselev_i_max_value_in_strings/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kiselev_i_max_value_in_strings {

KiselevITestTaskMPI::KiselevITestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool KiselevITestTaskMPI::ValidationImpl() {
  const auto &matrix =GetInput();
  GetOutput().resize(matrix.size());
  return true;
}

bool KiselevITestTaskMPI::PreProcessingImpl() {
  const auto &matrix =GetInput();
  if(matrix.empty()){
    return false;
  }
  return true;
}

bool KiselevITestTaskMPI::RunImpl() {
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

bool KiselevITestTaskMPI::PostProcessingImpl() {
  return true;
}

} 
