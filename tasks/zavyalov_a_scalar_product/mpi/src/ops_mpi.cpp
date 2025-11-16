#include "zavyalov_a_scalar_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>
#include <algorithm>
#include "zavyalov_a_scalar_product/common/include/common.hpp"

namespace zavyalov_a_scalar_product {

ZavyalovAScalarProductMPI::ZavyalovAScalarProductMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool ZavyalovAScalarProductMPI::ValidationImpl() {
  return (!std::get<0>(GetInput()).empty()) && (std::get<0>(GetInput()).size() == std::get<1>(GetInput()).size());
}

bool ZavyalovAScalarProductMPI::PreProcessingImpl() {
  return true;
}

bool ZavyalovAScalarProductMPI::RunImpl() {
  auto &input = GetInput();
  const std::vector<double> &left = std::get<0>(input);
  const std::vector<double> &right = std::get<1>(input);

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int vector_size = static_cast<int>(left.size());

  int blocksize = vector_size / world_size;
  int elements_left = vector_size - (world_size * blocksize);

  double cur_res = 0.0;
  int start = (blocksize * rank) + std::min(rank, elements_left);
  int end = start + blocksize + (rank < elements_left ? 1 : 0);
  for (int i = start; i < end; i++) {
    cur_res += left[i] * right[i];
  }

  double globRes = 0.0;
  MPI_Allreduce(&cur_res, &globRes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = globRes;
  return true;
}

bool ZavyalovAScalarProductMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_scalar_product
