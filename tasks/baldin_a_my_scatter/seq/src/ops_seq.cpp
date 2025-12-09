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
  const auto &[sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm] = GetInput();

  if (sendcount <= 0 || sendcount != recvcount || root < 0) {
    return false;
  }
  if (sendtype != recvtype) {
    return false;
  }

  auto is_sup_type = [](MPI_Datatype type) -> bool {
    return (type == MPI_INT || type == MPI_FLOAT || type == MPI_DOUBLE);
  };

  return is_sup_type(sendtype);
}

bool BaldinAMyScatterSEQ::PreProcessingImpl() {
  int root = std::get<6>(GetInput());

  int world_size = 0;
  MPI_Comm_size(std::get<7>(GetInput()), &world_size);

  // если root выходит за границы, корректируем его
  if (root >= world_size) {
    std::get<6>(GetInput()) = root % world_size;
  }

  return true;
}

bool BaldinAMyScatterSEQ::RunImpl() {
  auto &input = GetInput();
  const auto &[sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm] = input;
  MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);

  GetOutput() = recvbuf;
  return true;
}

bool BaldinAMyScatterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_my_scatter
