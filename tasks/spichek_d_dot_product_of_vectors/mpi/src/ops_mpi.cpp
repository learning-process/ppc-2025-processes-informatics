#include "spichek_d_dot_product_of_vectors/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "spichek_d_dot_product_of_vectors/common/include/common.hpp"
#include "util/include/util.hpp"

namespace spichek_d_dot_product_of_vectors {

SpichekDDotProductOfVectorsMPI::SpichekDDotProductOfVectorsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SpichekDDotProductOfVectorsMPI::ValidationImpl() {
  const auto &[vector1, vector2] = GetInput();
  return (!vector1.empty()) && (vector1.size() == vector2.size()) && (GetOutput() == 0);
}

bool SpichekDDotProductOfVectorsMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SpichekDDotProductOfVectorsMPI::RunImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &[vector1, vector2] = GetInput();

  // Всегда вычисляем только на root процессе
  int dot_product = 0;
  if (rank == 0) {
    if (vector1.size() != vector2.size() || vector1.empty()) {
      return false;
    }
    for (size_t i = 0; i < vector1.size(); ++i) {
      dot_product += vector1[i] * vector2[i];
    }
  }

  // Распространяем результат
  MPI_Bcast(&dot_product, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = dot_product;

  return true;
}
bool SpichekDDotProductOfVectorsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_dot_product_of_vectors
