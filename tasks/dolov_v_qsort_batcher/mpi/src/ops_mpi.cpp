#include "dolov_v_qsort_batcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>

#include "dolov_v_qsort_batcher/common/include/common.hpp"

namespace dolov_v_qsort_batcher {

DolovVQsortBatcherMPI::DolovVQsortBatcherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool DolovVQsortBatcherMPI::ValidationImpl() {
  return true;
}

bool DolovVQsortBatcherMPI::PreProcessingImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (world_rank == 0) {
    total_count_ = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&total_count_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_count_ < 0) {
    return false;
  }

  part_sizes_.assign(world_size, 0);
  part_offsets_.assign(world_size, 0);

  int base_size = total_count_ / world_size;
  int extra = total_count_ % world_size;
  int current_offset = 0;

  for (int i = 0; i < world_size; ++i) {
    part_sizes_[i] = base_size + (i < extra ? 1 : 0);
    part_offsets_[i] = current_offset;
    current_offset += part_sizes_[i];
  }

  local_buffer_.resize(part_sizes_[world_rank]);
  return true;
}

bool DolovVQsortBatcherMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (total_count_ <= 0) {
    return true;
  }

  DistributeData(world_rank, world_size);

  if (!local_buffer_.empty()) {
    FastSort(local_buffer_.data(), 0, static_cast<int>(local_buffer_.size()) - 1);
  }

  ExecuteBatcherParallel(world_rank, world_size);

  CollectData(world_rank, world_size);

  return true;
}

void DolovVQsortBatcherMPI::DistributeData(int world_rank, int /*world_size*/) {
  const double *send_ptr = (world_rank == 0) ? GetInput().data() : nullptr;
  MPI_Scatterv(send_ptr, part_sizes_.data(), part_offsets_.data(), MPI_DOUBLE, local_buffer_.data(),
               part_sizes_[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void DolovVQsortBatcherMPI::ExecuteBatcherParallel(int world_rank, int world_size) {
  if (world_size <= 1 || total_count_ == 0) {
    return;
  }

  std::vector<double> neighbor_data;

  for (int step = 0; step < world_size; ++step) {
    int partner = -1;

    if (step % 2 == 0) {
      if (world_rank % 2 == 0) {
        partner = world_rank + 1;
      } else {
        partner = world_rank - 1;
      }
    } else {
      if (world_rank % 2 != 0) {
        partner = world_rank + 1;
      } else {
        partner = world_rank - 1;
      }
    }

    if (partner >= 0 && partner < world_size) {
      int partner_size = part_sizes_[partner];
      neighbor_data.resize(partner_size);

      MPI_Sendrecv(local_buffer_.data(), static_cast<int>(local_buffer_.size()), MPI_DOUBLE, partner, 0,
                   neighbor_data.data(), partner_size, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<double> merged;
      MergeSequences(local_buffer_, neighbor_data, merged, world_rank < partner);
      local_buffer_ = std::move(merged);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void DolovVQsortBatcherMPI::MergeSequences(const std::vector<double> &first, const std::vector<double> &second,
                                           std::vector<double> &result, bool take_low) {
  size_t n1 = first.size();
  size_t n2 = second.size();
  std::vector<double> combined(n1 + n2);

  std::merge(first.begin(), first.end(), second.begin(), second.end(), combined.begin());

  result.resize(n1);
  if (take_low) {
    std::copy(combined.begin(), combined.begin() + n1, result.begin());
  } else {
    std::copy(combined.end() - n1, combined.end(), result.begin());
  }
}

void DolovVQsortBatcherMPI::CollectData(int world_rank, int /*world_size*/) {
  std::vector<double> global_array;
  if (world_rank == 0) {
    global_array.resize(total_count_);
  }

  MPI_Gatherv(local_buffer_.data(), static_cast<int>(local_buffer_.size()), MPI_DOUBLE, global_array.data(),
              part_sizes_.data(), part_offsets_.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    GetOutput() = std::move(global_array);
  }
}

int DolovVQsortBatcherMPI::GetSplitIndex(double *data, int low, int high) {
  double pivot = data[low + (high - low) / 2];
  int i = low - 1;
  int j = high + 1;
  while (true) {
    do {
      i++;
    } while (data[i] < pivot);
    do {
      j--;
    } while (data[j] > pivot);
    if (i >= j) {
      return j;
    }
    std::swap(data[i], data[j]);
  }
}

void DolovVQsortBatcherMPI::FastSort(double *data, int low, int high) {
  if (low < high) {
    int split_point = GetSplitIndex(data, low, high);
    FastSort(data, low, split_point);
    FastSort(data, split_point + 1, high);
  }
}

bool DolovVQsortBatcherMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dolov_v_qsort_batcher
