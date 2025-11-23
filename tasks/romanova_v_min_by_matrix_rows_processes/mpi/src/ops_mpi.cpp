#include "romanova_v_min_by_matrix_rows_processes/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
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
  return true;
}

bool RomanovaVMinByMatrixRowsMPI::RunImpl() {
  int n = 0;
  int rank = 0;
  int delta = 0;
  int extra = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &n);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> recv_counts;
  std::vector<int> send_counts;

  std::vector<int> displs_scatt;
  std::vector<int> displs_gath;

  OutType flat_data_;

  if (rank == 0) {
    n_ = GetInput().size();
    m_ = GetInput()[0].size();

    delta = static_cast<int>(n_ / n);
    extra = static_cast<int>(n_ % n);

    recv_counts = std::vector<int>(n, delta);
    recv_counts[n - 1] += extra;

    send_counts = std::vector<int>(n, delta * m_);
    send_counts[n - 1] += extra * m_;

    displs_gath = std::vector<int>(n);
    displs_scatt = std::vector<int>(n);

    for (int i = 1; i < n; i++) {
      displs_gath[i] = displs_gath[i - 1] + delta;
      displs_scatt[i] = displs_scatt[i - 1] + delta * m_;
    }

    for (const auto &vec : GetInput()) {
      flat_data_.insert(flat_data_.end(), vec.begin(), vec.end());
    }
  }

  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(&delta, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&extra, 1, MPI_INT, 0, MPI_COMM_WORLD);

  OutType local_data_((delta + (rank == n - 1 ? extra : 0)) * m_);

  MPI_Scatterv(flat_data_.data(), send_counts.data(), displs_scatt.data(), MPI_INT, local_data_.data(),
               local_data_.size(), MPI_INT, 0, MPI_COMM_WORLD);

  OutType temp_(delta + (rank == n - 1 ? extra : 0));

  for (int i = 0; i < temp_.size(); i++) {
    temp_[i] = local_data_[i * m_];
    for (size_t j = 1; j < m_; j++) {
      temp_[i] = std::min(temp_[i], local_data_[i * m_ + j]);
    }
  }

  res_ = OutType(n_);
  MPI_Gatherv(temp_.data(), static_cast<int>(temp_.size()), MPI_INT, res_.data(), recv_counts.data(),
              displs_gath.data(), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(res_.data(), static_cast<int>(n_), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool RomanovaVMinByMatrixRowsMPI::PostProcessingImpl() {
  GetOutput() = res_;
  return true;
}

}  // namespace romanova_v_min_by_matrix_rows_processes
