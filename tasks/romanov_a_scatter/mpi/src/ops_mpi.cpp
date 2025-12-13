#include "romanov_a_scatter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>

#include "romanov_a_scatter/common/include/common.hpp"

namespace romanov_a_scatter {

static inline int GetVirtualRank(int rank, int num_processes, int root) {
  return (rank - root + num_processes) % num_processes;
}

static inline int GetRealRank(int vrank, int num_processes, int root) {
  return (vrank + root) % num_processes;
}

static int ComputeSubtreeSize(int vrank, int num_processes) {
  int result = 0;
  while (vrank < num_processes) {
    result += 1 + ComputeSubtreeSize((vrank * 2) + 2, num_processes);
    vrank = (vrank * 2) + 1;
  }
  return result;
}

static void FillDataBuffer(int vrank, int num_processes, int root, char* data, char* src, int count, int type_size, int &offset) {
    if (vrank >= num_processes) {
      return;
    }
    
    int rank = GetRealRank(vrank, num_processes, root);

    int block_size = count * type_size;
    std::memcpy(data + offset, src + rank * block_size, block_size);
    
    offset += block_size;
    
    int left_vrank = 2 * vrank + 1;
    FillDataBuffer(left_vrank, num_processes, root, data, src, count, type_size, offset);
    
    int right_vrank = 2 * vrank + 2;
    FillDataBuffer(right_vrank, num_processes, root, data, src, count, type_size, offset);
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
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  
  int num_processes = 0;
  MPI_Comm_size(comm, &num_processes);

  int vrank = GetVirtualRank(rank, num_processes, root);
  
  int sendtype_size = 0;
  MPI_Type_size(sendtype, &sendtype_size);
  int recvtype_size = 0;
  MPI_Type_size(recvtype, &recvtype_size);

  int vrank_left = vrank * 2 + 1;
  int rank_left = -1;
  if (vrank_left < num_processes) {
    rank_left = GetRealRank(vrank_left, num_processes, root);
  }
  int left_subtree_sz = ComputeSubtreeSize(vrank_left, num_processes);

  int vrank_right = vrank * 2 + 2;
  int rank_right = -1;
  if (vrank_right < num_processes) {
    rank_right = GetRealRank(vrank_right, num_processes, root);
  }
  int right_subtree_sz = ComputeSubtreeSize(vrank_right, num_processes);

  int subtree_sz = left_subtree_sz + right_subtree_sz + 1;

  char* reordered_data = static_cast<char*>(malloc(subtree_sz * recvcount * recvtype_size));
  (void)sendcount;

  if (rank == root) {
    char* src = static_cast<char*>(sendbuf);
    int offset = 0;

    FillDataBuffer(vrank, num_processes, root, reordered_data, src, recvcount, recvtype_size, offset);
  }
  else {
    int vparent_rank = (vrank - 1) / 2;
    int parent_rank = GetRealRank(vparent_rank, num_processes, root);
    
    MPI_Recv(reordered_data, subtree_sz * recvcount, recvtype, parent_rank, 0, comm, MPI_STATUS_IGNORE);
  }

  std::memcpy(recvbuf, reordered_data, recvcount * recvtype_size);
  
  if (rank_left != -1) {
    char* left_data = reordered_data + recvcount * recvtype_size;
    MPI_Send(left_data, left_subtree_sz * recvcount, recvtype, rank_left, 0, comm);
  }
  
  if (rank_right != -1) {
    char* right_data = reordered_data + (1 + left_subtree_sz) * recvcount * recvtype_size;
    MPI_Send(right_data, right_subtree_sz * recvcount, recvtype, rank_right, 0, comm);
  }

  free(reordered_data);

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
