#include "zavyalov_a_reduce/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <cstdlib>
#include <vector>

#include "zavyalov_a_reduce/common/include/common.hpp"

namespace zavyalov_a_reduce {

ZavyalovAReduceSEQ::ZavyalovAReduceSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  std::get<0>(GetOutput()) = nullptr;
}

bool ZavyalovAReduceSEQ::ValidationImpl() {
  return true;
  /*
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (rank != 0) {
    return true;
  }
  bool res = true;
  MPI_Op operation = std::get<0>(GetInput());
  res &= (operation == MPI_SUM || operation == MPI_MIN ); // TODO Добавить все поддерживаемые операции
  MPI_Datatype cur_type = std::get<1>(GetInput());
  res &= (cur_type == MPI_INT || cur_type == MPI_FLOAT || cur_type == MPI_DOUBLE); // TODO Добавить все поддерживаемые
  типы данных size_t sz = std::get<2>(GetInput()); res &= sz > 0; void* mem = std::get<3>(GetInput()); res &= (mem !=
  nullptr); int receiver_rank = std::get<4>(GetInput()); res &= (receiver_rank < world_size); return res;
  */
}

bool ZavyalovAReduceSEQ::PreProcessingImpl() {
  return true;
}

bool ZavyalovAReduceSEQ::RunImpl() {
  std::get<1>(GetOutput()) = true;
  return true;
  MPI_Op operation = std::get<0>(GetInput());
  MPI_Datatype cur_type = std::get<1>(GetInput());
  size_t sz = std::get<2>(GetInput());
  void *mem = std::get<3>(GetInput());
  int receiver_rank = std::get<4>(GetInput());

  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::cout << "rank " << rank << " mem: ";
  for (size_t i = 0; i < sz; i++) {
    std::cout << ((int *)(mem))[i] << " ";
  }
  std::cout << std::endl;

  void *result_buf = new int[sz];  // не всегда int[], в общем случае T[]
  if (rank == receiver_rank) {
    MPI_Reduce(mem, result_buf, sz, cur_type, operation, receiver_rank, MPI_COMM_WORLD);
    MPI_Bcast(result_buf, sz, cur_type, receiver_rank, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(mem, nullptr, sz, cur_type, operation, receiver_rank, MPI_COMM_WORLD);
    MPI_Bcast(result_buf, sz, cur_type, receiver_rank, MPI_COMM_WORLD);
  }

  std::get<0>(GetOutput()) = result_buf;

  std::cout << "rank " << rank << " result_buf: ";
  for (size_t i = 0; i < sz; i++) {
    std::cout << ((int *)(result_buf))[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "rank " << rank << " std::get<0>(GetOutput()): ";
  for (size_t i = 0; i < sz; i++) {
    std::cout << ((int *)(std::get<0>(GetOutput())))[i] << " ";
  }
  std::cout << std::endl;
  return true;
}

bool ZavyalovAReduceSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_reduce
