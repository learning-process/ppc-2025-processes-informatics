#include "romanov_a_scatter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>

#include "romanov_a_scatter/common/include/common.hpp"

namespace romanov_a_scatter {

int MyMPI_Scatter(
  void         *sendbuf,
  int          sendcount,
  MPI_Datatype sendtype,
  void         *recvbuf,
  int          recvcount,
  MPI_Datatype recvtype,
  int          root,
  MPI_Comm     comm
) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_processes = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  int sendtype_size = 0;
  MPI_Type_size(sendtype, &sendtype_size);
  int recvtype_size = 0;
  MPI_Type_size(recvtype, &recvtype_size);

  if (rank == root) {
    std::memcpy(
      recvbuf,
      static_cast<char*>(sendbuf) + rank * sendcount * sendtype_size,
      sendcount * sendtype_size
    );

    for (int i = 0; i < num_processes; ++i) {
      if (i == root) {
        continue;
      }

      MPI_Send(
        static_cast<char*>(sendbuf) + i * sendcount * sendtype_size,
        sendcount,
        sendtype,
        i,
        0,
        comm
      );
    }
  } 
  else {
    MPI_Recv(
      recvbuf,
      recvcount,
      recvtype,
      root,
      0,
      comm,
      MPI_STATUS_IGNORE
    );
  }

  return MPI_SUCCESS;
}

RomanovAScatterMPI::RomanovAScatterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>{};
}

bool RomanovAScatterMPI::ValidationImpl() {
  return true;
}

bool RomanovAScatterMPI::PreProcessingImpl() {
  return true;
}

bool RomanovAScatterMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_processes = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  int root = std::get<2>(GetInput());
  int sendcount = std::get<1>(GetInput());

  std::vector<int> sendbuf;
  if (rank == root) {
    sendbuf = std::get<0>(GetInput());
    sendbuf.resize(num_processes * sendcount);
  }

  std::vector<int> recvbuf(sendcount);

  MyMPI_Scatter(
    sendbuf.data(),
    sendcount,
    MPI_INT,
    recvbuf.data(),
    sendcount,
    MPI_INT,
    root,
    MPI_COMM_WORLD
  );

  GetOutput() = recvbuf;

  return true;
}

bool RomanovAScatterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_a_scatter
