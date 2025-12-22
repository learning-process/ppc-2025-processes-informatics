#include "yurkin_g_ruler/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <vector>

#include "util/include/util.hpp"
#include "yurkin_g_ruler/common/include/common.hpp"

namespace yurkin_g_ruler {

YurkinGRulerMPI::YurkinGRulerMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinGRulerMPI::ValidationImpl() {
  return (GetInput() >= 0) && (GetOutput() == 0);
}

bool YurkinGRulerMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

namespace {
int ParseEnvOrDefault(const char *name, int def) {
  const char *val = std::getenv(name);
  if (val == nullptr) {
    return def;
  }
  char *end = nullptr;
  errno = 0;
  long v = std::strtol(val, &end, 10);
  if (end == val || errno != 0) {
    return def;
  }
  if (v < INT_MIN) {
    return INT_MIN;
  }
  if (v > INT_MAX) {
    return INT_MAX;
  }
  return static_cast<int>(v);
}
}  // namespace

bool YurkinGRulerMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size <= 0) {
    return false;
  }

  int default_src = 0;
  int default_dst = (size > 0) ? (size - 1) : 0;
  int src = ParseEnvOrDefault("PPC_SRC", default_src);
  int dst = ParseEnvOrDefault("PPC_DST", default_dst);

  if (src < 0) {
    src = 0;
  }
  if (src >= size) {
    src = size - 1;
  }
  if (dst < 0) {
    dst = 0;
  }
  if (dst >= size) {
    dst = size - 1;
  }

  int payload = GetInput();

  if (src == dst) {
    if (rank == src) {
      GetOutput() = payload;
    } else {
      GetOutput() = 0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  int direction = (dst > src) ? +1 : -1;

  bool on_path = false;
  if (direction > 0) {
    on_path = (rank >= src && rank <= dst);
  } else {
    on_path = (rank <= src && rank >= dst);
  }

  if (!on_path) {
    GetOutput() = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  if (rank == src) {
    int next = rank + direction;
    MPI_Send(&payload, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    GetOutput() = 0;
  } else {
    int prev = rank - direction;
    int recv_val = 0;
    MPI_Recv(&recv_val, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank == dst) {
      GetOutput() = recv_val;
    } else {
      int next = rank + direction;
      MPI_Send(&recv_val, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
      GetOutput() = 0;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool YurkinGRulerMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace yurkin_g_ruler
