#include "leonova_a_most_diff_neigh_vec_elems/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"
#include "util/include/util.hpp"

namespace leonova_a_most_diff_neigh_vec_elems {

LeonovaAMostDiffNeighVecElemsMPI::LeonovaAMostDiffNeighVecElemsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::tuple<int, int>(0, 0);
}

bool LeonovaAMostDiffNeighVecElemsMPI::ValidationImpl() {
  return true;
}

bool LeonovaAMostDiffNeighVecElemsMPI::PreProcessingImpl() {
  return true;
}

bool LeonovaAMostDiffNeighVecElemsMPI::RunImpl() {
  // auto input = GetInput();
  // if (input == 0) {
  //   return false;
  // }

  // for (InType i = 0; i < GetInput(); i++) {
  //   for (InType j = 0; j < GetInput(); j++) {
  //     for (InType k = 0; k < GetInput(); k++) {
  //       std::vector<InType> tmp(i + j + k, 1);
  //       GetOutput() += std::accumulate(tmp.begin(), tmp.end(), 0);
  //       GetOutput() -= i + j + k;
  //     }
  //   }
  // }

  // const int num_threads = ppc::util::GetNumThreads();
  // GetOutput() *= num_threads;

  // int rank = 0;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // if (rank == 0) {
  //   GetOutput() /= num_threads;
  // } else {
  //   int counter = 0;
  //   for (int i = 0; i < num_threads; i++) {
  //     counter++;
  //   }

  //   if (counter != 0) {
  //     GetOutput() /= counter;
  //   }
  // }

  // MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool LeonovaAMostDiffNeighVecElemsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace leonova_a_most_diff_neigh_vec_elems
