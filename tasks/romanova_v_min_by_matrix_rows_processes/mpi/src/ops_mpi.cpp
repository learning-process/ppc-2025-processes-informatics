#include "romanova_v_min_by_matrix_rows_processes/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "romanova_v_min_by_matrix_rows_processes/common/include/common.hpp"

namespace romanova_v_min_by_matrix_rows_processes {

RomanovaVMinByMatrixRowsMPI::RomanovaVMinByMatrixRowsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType(in.size());
}

bool RomanovaVMinByMatrixRowsMPI::ValidationImpl() {
  return !GetInput().empty() && !GetInput()[0].empty();
}

bool RomanovaVMinByMatrixRowsMPI::PreProcessingImpl() {
  in_data_ = GetInput();
  n_ = in_data_.size();
  m_ = in_data_[0].size();
  res_ = OutType(n_);
  return true;
}

bool RomanovaVMinByMatrixRowsMPI::RunImpl() {
  int n, rank;
  int delta, extra = 0;
  std::vector<int> recv_counts, displs;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    delta = n_ / n;
    extra = n_ % n;

    recv_counts = std::vector<int>(n, delta);
    recv_counts[n - 1] += extra;

    displs = std::vector<int>(n);
    for (int i = 1; i < n; i++) {
      displs[i] = displs[i - 1] + delta;
    }
  }
  MPI_Bcast(&delta, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&extra, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int st_row = rank * delta;
  int en_row = (rank + 1) * delta;
  if (rank == n - 1) {
    en_row += extra;
  }

  OutType temp(en_row - st_row);

  int min_val = 0;

  for (int i = 0; i < en_row - st_row; i++) {
    min_val = in_data_[st_row + i][0];
    for (int j = 1; j < m_; j++) {
      min_val = std::min(min_val, in_data_[st_row + i][j]);
    }
    temp[i] = min_val;
  }

  MPI_Gatherv(temp.data(), temp.size(), MPI_INT, res_.data(), recv_counts.data(), displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);
  MPI_Bcast(res_.data(), n_, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool RomanovaVMinByMatrixRowsMPI::PostProcessingImpl() {
  GetOutput() = res_;
  return true;
}

}  // namespace romanova_v_min_by_matrix_rows_processes
