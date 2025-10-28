#include "levonychev_i_min_val_rows_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "levonychev_i_min_val_rows_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_min_val_rows_matrix {

LevonychevIMinValRowsMatrixMPI::LevonychevIMinValRowsMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}
bool LevonychevIMinValRowsMatrixMPI::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }
  size_t row_length = GetInput()[0].size();
  for (size_t i = 1; i < GetInput().size(); ++i) {
    if (GetInput()[i].size() != row_length) {
      return false;
    }
  }
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::PreProcessingImpl() {
  GetOutput().resize(GetInput().size());
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }
  const InType& matrix = GetInput();
  OutType& min_values = GetOutput();

  for (size_t i = 0; i < matrix.size(); ++i)
  { 
    double min_val = matrix[i][0];
    for (size_t j = 1; j < matrix[i].size(); ++j)
    {
      if (matrix[i][j] < min_val)
      {
        min_val = matrix[i][j];
      }
    }
    min_values[i] = min_val;
  }

  return true;
}

bool LevonychevIMinValRowsMatrixMPI::PostProcessingImpl() {
  return GetInput().size() == GetOutput().size();
}

}  // namespace levonychev_i_min_val_rows_matrix
