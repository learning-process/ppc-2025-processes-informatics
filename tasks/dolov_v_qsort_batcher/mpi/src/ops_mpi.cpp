#include "dolov_v_qsort_batcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace dolov_v_qsort_batcher {

DolovVQsortBatcherMPI::DolovVQsortBatcherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool DolovVQsortBatcherMPI::ValidationImpl() {
  return true;
}

bool DolovVQsortBatcherMPI::PreProcessingImpl() {
  int rank, proc_count;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  if (rank == 0) {
    total_elements_ = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&total_elements_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  send_counts_.resize(proc_count);
  displacements_.resize(proc_count);
  SetupWorkload(total_elements_, proc_count, send_counts_, displacements_);

  local_data_.resize(send_counts_[rank]);
  return true;
}

bool DolovVQsortBatcherMPI::RunImpl() {
  int rank, proc_count;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  if (total_elements_ <= 0) {
    return true;
  }

  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, send_counts_.data(), displacements_.data(), MPI_DOUBLE,
               local_data_.data(), send_counts_[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (!local_data_.empty()) {
    std::sort(local_data_.begin(), local_data_.end());
  }

  RunBatcherMerge(rank, proc_count, local_data_);

  std::vector<double> global_res;
  if (rank == 0) {
    global_res.resize(total_elements_);
  }

  MPI_Gatherv(local_data_.data(), static_cast<int>(local_data_.size()), MPI_DOUBLE, global_res.data(),
              send_counts_.data(), displacements_.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = std::move(global_res);
  }

  return true;
}

void DolovVQsortBatcherMPI::SetupWorkload(int total_size, int proc_count, std::vector<int> &counts,
                                          std::vector<int> &offsets) {
  int base = total_size / proc_count;
  int rem = total_size % proc_count;
  for (int i = 0; i < proc_count; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    offsets[i] = (i == 0) ? 0 : offsets[i - 1] + counts[i - 1];
  }
}

void DolovVQsortBatcherMPI::RunBatcherMerge(int rank, int proc_count, std::vector<double> &local_vec) {
  for (int phase = 0; phase < proc_count; ++phase) {
    int partner;
    if (phase % 2 == 0) {
      // Четная фаза: пары (0,1), (2,3)...
      partner = (rank % 2 == 0) ? rank + 1 : rank - 1;
    } else {
      // Нечетная фаза: пары (1,2), (3,4)...
      partner = (rank % 2 != 0) ? rank + 1 : rank - 1;
    }

    if (partner >= 0 && partner < proc_count) {
      ExchangeAndMerge(partner, local_vec, rank < partner);
    }
  }
}

void DolovVQsortBatcherMPI::ExchangeAndMerge(int partner, std::vector<double> &local_vec, bool keep_smaller) {
  int local_size = static_cast<int>(local_vec.size());
  int partner_size;

  MPI_Sendrecv(&local_size, 1, MPI_INT, partner, 0, &partner_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  if (local_size == 0 && partner_size == 0) {
    return;
  }

  std::vector<double> remote_vec(partner_size);
  MPI_Sendrecv(local_vec.data(), local_size, MPI_DOUBLE, partner, 1, remote_vec.data(), partner_size, MPI_DOUBLE,
               partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<double> merged(local_size + partner_size);
  std::merge(local_vec.begin(), local_vec.end(), remote_vec.begin(), remote_vec.end(), merged.begin());

  if (keep_smaller) {
    local_vec.assign(merged.begin(), merged.begin() + local_size);
  } else {
    local_vec.assign(merged.begin() + partner_size, merged.end());
  }
}

bool DolovVQsortBatcherMPI::PostProcessingImpl() {
  local_data_.clear();
  send_counts_.clear();
  displacements_.clear();
  return true;
}

}  // namespace dolov_v_qsort_batcher
