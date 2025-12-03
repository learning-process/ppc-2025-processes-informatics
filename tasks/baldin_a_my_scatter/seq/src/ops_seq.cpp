#include "baldin_a_my_scatter/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "baldin_a_my_scatter/common/include/common.hpp"
#include "util/include/util.hpp"

namespace baldin_a_my_scatter {

BaldinAMyScatterSEQ::BaldinAMyScatterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BaldinAMyScatterSEQ::ValidationImpl() {
  const auto& input = GetInput();

  const auto& [sendbuf, sendcount, sendtype, 
                 recvbuf, recvcount, recvtype, 
                 root, comm] = input;

  int world_size = 0;
  MPI_Comm_size(comm, &world_size);

  auto is_sup_type = [](MPI_Datatype type) -> bool {
    return (type == MPI_INT || type == MPI_FLOAT || type == MPI_DOUBLE);
  };

  return (sendcount > 0 && 
            sendcount == recvcount && 
            sendtype == recvtype &&
            is_sup_type(sendtype) && 
            root >= 0);
}

bool BaldinAMyScatterSEQ::PreProcessingImpl() {
  auto& input = GetInput();
    
  auto& [sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm] = input;
  
  int world_size = 0;
  MPI_Comm_size(comm, &world_size);

  if (root >= world_size) {
      std::get<6>(input) = root % world_size; 
  }

  // int total_cnt = world_size * sendcount;
  // if (sendtype == MPI_INT) {
  //   std::vector<int> tmp(total_cnt);
  //   for (int i = 0; i < total_cnt; i++) {
  //     tmp[i] = i;
  //   }
  //   (int*)sendbuf = tmp.data();
  //   tmp.data() = nullptr;
  // }

  return true;
}

bool BaldinAMyScatterSEQ::RunImpl() {
  auto& input = GetInput();
  const auto& [sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm] = input;
  MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);

  GetOutput() = recvbuf;
  return true;
}

bool BaldinAMyScatterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_my_scatter
