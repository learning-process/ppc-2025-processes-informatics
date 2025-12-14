#include "romanov_a_scatter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstring>

#include "romanov_a_scatter/common/include/common.hpp"

namespace romanov_a_scatter {

static inline int GetVirtualRank(int rank, int num_processes, int root) {
  return (rank - root + num_processes) % num_processes;
}

static inline int GetRealRank(int vrank, int num_processes, int root) {
  return (vrank + root) % num_processes;
}

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

  int vrank = GetVirtualRank(rank, num_processes, root);

  char* data = nullptr;
  int blocks_in_subtree = 0;

  if (rank == root) {
    data = static_cast<char*>(sendbuf);
    blocks_in_subtree = num_processes;
  }
  else {
    int parent_vrank = vrank & (vrank - 1);
    int parent_rank = GetRealRank(parent_vrank, num_processes, root);
    MPI_Recv(&blocks_in_subtree, 1, MPI_INT, parent_rank, 0, comm, MPI_STATUS_IGNORE);

    if (blocks_in_subtree == 1) {
        MPI_Recv(recvbuf, blocks_in_subtree * recvcount, recvtype, parent_rank, 1, comm, MPI_STATUS_IGNORE);
    }
    else {
      data = static_cast<char*>(malloc(blocks_in_subtree * recvcount * recvtype_size));
      MPI_Recv(data, blocks_in_subtree * recvcount, recvtype, parent_rank, 1, comm, MPI_STATUS_IGNORE);
    }
  }

  // Есть блоки данных для дальшейшей передачи
  if (blocks_in_subtree > 1) {

    std::memcpy(recvbuf, data, recvcount * recvtype_size);

    int mask = 1;
    int vrank_copy = vrank;
    while (!(vrank_copy & 1) && mask < blocks_in_subtree) {
      mask <<= 1;
      vrank_copy >>= 1;
    }
    mask >>= 1;

    while (mask > 0) {
      int child_vrank = vrank | mask;
      if (child_vrank < num_processes) {
        int child_rank = GetRealRank(child_vrank, num_processes, root);
        int send_blocks = blocks_in_subtree - mask;
        MPI_Send(&send_blocks, 1, MPI_INT, child_rank, 0, comm);
        MPI_Send(data + mask * recvcount * recvtype_size, send_blocks * recvcount, recvtype, child_rank, 1, comm);
        blocks_in_subtree = mask;
      }
      mask >>= 1;
    }

    if (rank != root) {
      free(data);
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
