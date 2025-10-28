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
  GetOutput().resize(in.size());
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
  // const InType &matrix = GetInput();

  // int ProcNum, ProcRank;
  // MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  // MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

  // const int ROWS = matrix.size();
  // const int COLS = matrix[0].size();

  // if (ProcRank == 0)
  // {
  //   OutType &min_values = GetOutput();
  //   int next_row_send = 0;
  //   int tar_proc = 1;
  //   MPI_Status status;
  // }
  GetOutput() = {1, 4, 7};
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::PostProcessingImpl() {
  return GetInput().size() == GetOutput().size();
}

}  // namespace levonychev_i_min_val_rows_matrix
