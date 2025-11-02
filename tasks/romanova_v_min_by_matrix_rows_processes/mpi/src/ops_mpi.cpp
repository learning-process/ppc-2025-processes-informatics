#include "romanova_v_min_by_matrix_rows_processes/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "romanova_v_min_by_matrix_rows_processes/common/include/common.hpp"
#include "util/include/util.hpp"

namespace romanova_v_min_by_matrix_rows_processes {

RomanovaVMinByMatrixRowsMPI::RomanovaVMinByMatrixRowsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>(in.size());
}

bool RomanovaVMinByMatrixRowsMPI::ValidationImpl() {
  return GetInput().size() != 0 && GetInput()[0].size() >= 1;
}

bool RomanovaVMinByMatrixRowsMPI::PreProcessingImpl() {
  in_data_ = GetInput();
  n_ = in_data_.size();
  m_ = in_data_[0].size();
  res_ = std::vector<int>(n_);
  return true;
}

bool RomanovaVMinByMatrixRowsMPI::RunImpl() {
  return false;
}

bool RomanovaVMinByMatrixRowsMPI::PostProcessingImpl() {
  GetOutput() = res_;
  return true;
}

}  // namespace romanova_v_min_by_matrix_rows_processes
