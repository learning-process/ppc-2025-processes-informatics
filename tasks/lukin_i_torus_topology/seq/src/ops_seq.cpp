#include "lukin_i_torus_topology/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "lukin_i_torus_topology/common/include/common.hpp"

namespace lukin_i_torus_topology {

LukinIThorTopologySEQ::LukinIThorTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  OutType out = std::make_tuple(std::vector<int>{}, std::vector<int>{});
  GetOutput() = out;
}

bool LukinIThorTopologySEQ::ValidationImpl() {
  int start_v = std::get<0>(GetInput());
  int end_v = std::get<1>(GetInput());

  if (start_v < 0 || end_v < 0) {
    return false;
  }

  int proc_count = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
  if (start_v > proc_count - 1 || end_v > proc_count - 1) {
    return false;
  }

  return true;
}

bool LukinIThorTopologySEQ::PreProcessingImpl() {
  start = std::get<0>(GetInput());
  end = std::get<1>(GetInput());
  return true;
}

bool LukinIThorTopologySEQ::RunImpl() {
  int rank = -1;
  int proc_count = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  std::vector<int> message;
  int message_len = -1;
  if (rank == start) {
    message = std::get<2>(GetInput());
    message_len = message.size();
  }

  if (proc_count == 1) {
    GetOutput() = std::make_tuple(std::vector<int>{}, message);
    return true;
  }

  if (start == end) {
    MPI_Bcast(&message_len, 1, MPI_INT, end, MPI_COMM_WORLD);
    message.resize(message_len);
    MPI_Bcast(message.data(), message_len, MPI_INT, end, MPI_COMM_WORLD);
    GetOutput() = std::make_tuple(std::vector<int>{}, message);
    return true;
  }

  int cols = -1;
  int rows = -1;
  for (rows = sqrt(proc_count); rows > 0; rows--) {
    if (proc_count % rows == 0) {
      cols = proc_count / rows;
      break;
    }
  }

  int dims[2] = {rows, cols};
  int periods[2] = {1, 1};

  MPI_Comm MPI_COMM_TORUS;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &MPI_COMM_TORUS);

  if (rank == start) {
    MPI_Send(&message_len, 1, MPI_INT, end, 0, MPI_COMM_TORUS);
    MPI_Send(message.data(), message_len, MPI_INT, end, 1, MPI_COMM_TORUS);
  } else if (rank == end) {
    MPI_Recv(&message_len, 1, MPI_INT, start, 0, MPI_COMM_TORUS, MPI_STATUS_IGNORE);
    message.resize(message_len);
    MPI_Recv(message.data(), message_len, MPI_INT, start, 1, MPI_COMM_TORUS, MPI_STATUS_IGNORE);
  }

  MPI_Bcast(&message_len, 1, MPI_INT, end, MPI_COMM_WORLD);
  message.resize(message_len);
  MPI_Bcast(message.data(), message_len, MPI_INT, end, MPI_COMM_WORLD);

  OutType out = std::make_tuple(dummy_route, message);
  GetOutput() = out;

  MPI_Comm_free(&MPI_COMM_TORUS);

  return true;
}

bool LukinIThorTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace lukin_i_torus_topology
