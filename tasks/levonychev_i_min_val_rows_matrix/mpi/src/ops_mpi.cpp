#include "levonychev_i_min_val_rows_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "levonychev_i_min_val_rows_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_min_val_rows_matrix {

LevonychevIMinValRowsMatrixMPI::LevonychevIMinValRowsMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  for (size_t i = 0; i < in.size(); ++i) {
    GetInput().push_back(in[i]);
  }
  // GetInput() = in;
  GetOutput().resize(in.size());
}
bool LevonychevIMinValRowsMatrixMPI::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }
  // size_t row_length = GetInput()[0].size();
  // for (size_t i = 1; i < GetInput().size(); ++i) {
  //   if (GetInput()[i].size() != row_length) {
  //     return false;
  //   }
  // }
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
  const InType &matrix = GetInput();
  OutType &min_values = GetOutput();

  int ProcNum, ProcRank;
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

  const int ROWS = matrix.size();
  const int COLS = matrix[0].size();

  if (ProcRank == 0) {
    int target_proc = 1;
    for (int i = 0; i < ROWS; ++i) {
      if (target_proc == ProcNum) {
        target_proc = 1;
      }
      MPI_Send(matrix[i].data(), COLS, MPI_DOUBLE, target_proc, i, MPI_COMM_WORLD);
      target_proc++;
    }

    for (int i = 1; i < ProcNum; ++i) {
      MPI_Send(nullptr, 0, MPI_DOUBLE, i, ROWS, MPI_COMM_WORLD);
    }
  }
  if (ProcRank > 0) {
    OutType local_row;
    local_row.resize(COLS);
    int row_tag;
    MPI_Status status;

    while (true) {
      MPI_Recv(local_row.data(), COLS, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      row_tag = status.MPI_TAG;

      if (row_tag == ROWS) {
        break;
      }

      double min_value;
      if (local_row.empty()) {
        min_value = std::numeric_limits<double>::quiet_NaN();
      } else {
        min_value = *std::min_element(local_row.begin(), local_row.end());
      }

      MPI_Send(&min_value, 1, MPI_DOUBLE, 0, row_tag, MPI_COMM_WORLD);
    }
  }
  if (ProcRank == 0) {
    MPI_Status status;
    double rec_min_val;
    for (int i = 0; i < ROWS; ++i) {
      MPI_Recv(&rec_min_val, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      int row_index = status.MPI_TAG;
      min_values[row_index] = rec_min_val;
    }
    // Возможно так нельзя, но не проходят тесты для процессов 1 - 3
    for (int i = 1; i < ProcNum; ++i) {
      MPI_Send(min_values.data(), ROWS, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
  }
  if (ProcRank > 0) {
    MPI_Status status;
    MPI_Recv(min_values.data(), ROWS, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  }
  // GetOutput() = {1, 4, 7};
  return true;
}

bool LevonychevIMinValRowsMatrixMPI::PostProcessingImpl() {
  return GetInput().size() == GetOutput().size();
}

}  // namespace levonychev_i_min_val_rows_matrix
