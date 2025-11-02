#include "romanova_v_min_by_matrix_rows_processes/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>
#include <limits>

#include <iostream>

#include "romanova_v_min_by_matrix_rows_processes/common/include/common.hpp"
#include "util/include/util.hpp"

namespace romanova_v_min_by_matrix_rows_processes {

RomanovaVMinByMatrixRowsSEQ::RomanovaVMinByMatrixRowsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>(in.size());
}

bool RomanovaVMinByMatrixRowsSEQ::ValidationImpl() {
  return GetInput().size() != 0 && GetInput()[0].size() >= 1;
}

bool RomanovaVMinByMatrixRowsSEQ::PreProcessingImpl() {
  in_data_ = GetInput();
  n_ = in_data_.size();
  m_ = in_data_[0].size();
  res_ = std::vector<int>(n_);
  return true;
}

bool RomanovaVMinByMatrixRowsSEQ::RunImpl() {
  int min_val;
  for(int i = 0; i < n_; i++){
    min_val = INT_MAX;
    for(int j = 0; j < m_; j++){
      if(in_data_[i][j] < min_val) min_val = in_data_[i][j];
    }
    res_[i] = min_val;
  }
  return true;
}

bool RomanovaVMinByMatrixRowsSEQ::PostProcessingImpl() {
  GetOutput() = res_;
  return true;
}

}  // namespace romanova_v_min_by_matrix_rows_processes
