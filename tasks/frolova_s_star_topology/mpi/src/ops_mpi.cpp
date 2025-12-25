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
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    dest_ = GetInput();
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

    // Получаем все запросы и маршрутируем
    for (int i = 0; i < nodes; i++) {
      int dst = 0;
      MPI_Recv(&dst, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      const int src = status.MPI_SOURCE;

      if (dst < 0 || dst >= size) {
        int error = -1;
        MPI_Send(&error, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
        continue;
      }

      MPI_Probe(src, 0, MPI_COMM_WORLD, &status);
      int buf_size = 0;
      MPI_Get_count(&status, MPI_INT, &buf_size);
      buf.resize(buf_size);
      MPI_Recv(buf.data(), buf_size, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Отправляем получателю
      MPI_Send(&src, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
      MPI_Send(buf.data(), buf_size, MPI_INT, dst, 0, MPI_COMM_WORLD);
    }

    // Отправляем завершающие сообщения
    for (int i = 0; i < nodes; i++) {
      int finish = -1;
      MPI_Send(&finish, 1, MPI_INT, i + 1, 1, MPI_COMM_WORLD);
    }

  } else {
    // Отправляем запрос центральному
    MPI_Send(&dest_, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    // Отправляем данные (если есть)
    if (!data_.empty()) {
      MPI_Send(data_.data(), static_cast<int>(data_.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
      int dummy = 0;
      MPI_Send(&dummy, 0, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // Ждем сообщения от центрального
    while (true) {
      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      if (status.MPI_TAG == 1) {
        int finish = 0;
        MPI_Recv(&finish, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        break;
      } else if (status.MPI_TAG == 0) {
        int src = 0;
        MPI_Recv(&src, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (src < 0) {
          break;
        }

        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
        int buf_size = 0;
        MPI_Get_count(&status, MPI_INT, &buf_size);
        output_.resize(buf_size);
        MPI_Recv(output_.data(), buf_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
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
