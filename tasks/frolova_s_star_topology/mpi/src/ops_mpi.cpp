#include "frolova_s_star_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "frolova_s_star_topology/common/include/common.hpp"
// #include "util/include/util.hpp"

constexpr int kTerm = -1;  // terminating parameter

namespace frolova_s_star_topology {

FrolovaSStarTopologyMPI::FrolovaSStarTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool FrolovaSStarTopologyMPI::ValidationImpl() {
  // int rank = 0;
  // int size = 0;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // MPI_Comm_size(MPI_COMM_WORLD, &size);
  // return size >= 2 && (rank == 0 || ((GetInput() > 0)));
  return true;
}

bool FrolovaSStarTopologyMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank != 0) {
    if (size <= 1) {
      dest_ = 0;
    } else {
      dest_ = GetInput();

      while (dest_ >= size) {
        dest_ = dest_ - (size - 1);
      }

      // dest_ не может быть 0 (центральный процесс)
      // и не может быть равен rank (самому себе)
      if (dest_ == 0) {
        dest_ = 1;
      }

      if (dest_ == rank) {
        dest_ = (dest_ % (size - 1)) + 1;
        if (dest_ == rank) {
          dest_ = 1;
        }
      }
    }

    output_.resize(GetInput());
  }

  return true;
}

bool FrolovaSStarTopologyMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto nodes = size - 1;
  MPI_Status status;

  if (rank == 0) {
    std::vector<int> buf;
    for (int i = 0; i < nodes; i++) {
      int dst = 0;
      MPI_Recv(&dst, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      const int src = status.MPI_SOURCE;

      if (dst < 1 || dst >= size || dst == src) {
        int error = kTerm;
        MPI_Send(&error, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
        continue;
      }
      MPI_Probe(src, 0, MPI_COMM_WORLD, &status);
      int buf_size = 0;
      MPI_Get_count(&status, MPI_INT, &buf_size);
      buf.resize(buf_size);
      MPI_Recv(buf.data(), buf_size, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&src, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
      MPI_Send(buf.data(), buf_size, MPI_INT, dst, 0, MPI_COMM_WORLD);
    }
    for (int i = 0; i < nodes; i++) {
      MPI_Send(&kTerm, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
    }

  } else {
    MPI_Send(&dest_, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    if (!data_.empty()) {
      MPI_Send(data_.data(), static_cast<int>(data_.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
      int dummy = 0;
      MPI_Send(&dummy, 0, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    while (true) {
      int src = 0;
      MPI_Recv(&src, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (src == kTerm) {
        break;
      }
      MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
      int buf_size = 0;
      MPI_Get_count(&status, MPI_INT, &buf_size);
      output_.resize(buf_size);
      MPI_Recv(output_.data(), buf_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  return true;
}

bool FrolovaSStarTopologyMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    GetOutput() = static_cast<OutType>(output_.size());
  }
  return true;
}

}  // namespace frolova_s_star_topology
