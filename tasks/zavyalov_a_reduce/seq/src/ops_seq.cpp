#include "zavyalov_a_reduce/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <cstdlib>

#include "zavyalov_a_reduce/common/include/common.hpp"

namespace zavyalov_a_reduce {

ZavyalovAReduceSEQ::ZavyalovAReduceSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  std::get<0>(GetOutput()) = nullptr;
}

bool ZavyalovAReduceSEQ::ValidationImpl() {
  return true;
}

bool ZavyalovAReduceSEQ::PreProcessingImpl() {
  return true;
}

bool ZavyalovAReduceSEQ::RunImpl() {
  std::get<1>(GetOutput()) = true;
  std::get<0>(GetOutput()) = nullptr;
  return true;
  /* MPI_Op operation = std::get<0>(GetInput());
  MPI_Datatype cur_type = std::get<1>(GetInput());
  size_t sz = std::get<2>(GetInput());
  void *mem = std::get<3>(GetInput());
  int receiver_rank = std::get<4>(GetInput());

  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int type_size = 0;
  MPI_Type_size(cur_type, &type_size);
  void *result_buf = new char[sz * type_size];
  if (rank == receiver_rank) {
    MPI_Reduce(mem, result_buf, sz, cur_type, operation, receiver_rank, MPI_COMM_WORLD);
    MPI_Bcast(result_buf, static_cast<int>(sz), cur_type, receiver_rank, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(mem, nullptr, sz, cur_type, operation, receiver_rank, MPI_COMM_WORLD);
    MPI_Bcast(result_buf, static_cast<int>(sz), cur_type, receiver_rank, MPI_COMM_WORLD);
  }

  std::get<0>(GetOutput()) = result_buf;

  return true; */
}

bool ZavyalovAReduceSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_reduce
