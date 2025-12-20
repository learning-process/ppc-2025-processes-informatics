#include "romanov_a_scatter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstring>
#include <vector>

#include "romanov_a_scatter/common/include/common.hpp"

namespace romanov_a_scatter {

int MyMPIScatter(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                 MPI_Datatype recvtype, int root, MPI_Comm comm) {
  int sendtype_size = 0;
  MPI_Type_size(sendtype, &sendtype_size);

  int recvtype_size = 0;
  MPI_Type_size(recvtype, &recvtype_size);

  if (sendcount * sendtype_size < recvcount * recvtype_size) {
    return MPI_ERR_ARG;
  }

  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  int num_processes = 0;
  MPI_Comm_size(comm, &num_processes);

  if (rank == root) {
    int total_blocks = num_processes;
    MPI_Send(&total_blocks, 1, MPI_INT, 0, 0, comm);
    MPI_Send(sendbuf, total_blocks * recvcount * recvtype_size, MPI_BYTE, 0, 1, comm);
  }

  int parent_rank = rank & (rank - 1);

  int blocks_in_subtree = 0;
  MPI_Recv(&blocks_in_subtree, 1, MPI_INT, parent_rank, 0, comm, MPI_STATUS_IGNORE);

  if (blocks_in_subtree == 1) {
    MPI_Recv(recvbuf, recvcount * recvtype_size, MPI_BYTE, parent_rank, 1, comm, MPI_STATUS_IGNORE);
  } else {
    std::vector<char> data(static_cast<size_t>(blocks_in_subtree) * recvcount * recvtype_size);
    MPI_Recv(data.data(), blocks_in_subtree * recvcount * recvtype_size, MPI_BYTE, parent_rank, 1, comm,
             MPI_STATUS_IGNORE);

    std::memcpy(recvbuf, data.data(), recvcount * recvtype_size);

    int mask = 1;
    int rank_copy = rank;
    while (((rank_copy & 1) == 0) && (mask < blocks_in_subtree)) {
      mask <<= 1;
      rank_copy >>= 1;
    }
    mask >>= 1;

    // mask содержит такой наибольший бит, что все биты правее него (включая его) равны нулю

    while (mask > 0) {
      int child_rank = rank | mask;
      if (child_rank < num_processes) {
        int send_blocks = blocks_in_subtree - mask;
        MPI_Send(&send_blocks, 1, MPI_INT, child_rank, 0, comm);
        MPI_Send(data.data() + (mask * recvcount * recvtype_size), send_blocks * recvcount, recvtype, child_rank, 1,
                 comm);
        blocks_in_subtree = mask;
      }
      mask >>= 1;
    }
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

  // Эти данные должны быть известны каждому процессу для Scatter
  int root = std::get<2>(GetInput());
  int sendcount = std::get<1>(GetInput());

  std::vector<int> sendbuf;
  if (rank == root) {
    sendbuf = std::get<0>(GetInput());
    sendbuf.resize(num_processes * sendcount);
  }

  std::vector<int> recvbuf(sendcount);

  MyMPIScatter(sendbuf.data(), sendcount, MPI_INT, recvbuf.data(), sendcount, MPI_INT, root, MPI_COMM_WORLD);

  GetOutput() = recvbuf;

  return true;
}

bool RomanovAScatterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_a_scatter
