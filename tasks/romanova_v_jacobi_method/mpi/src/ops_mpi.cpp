#include "romanova_v_jacobi_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "romanova_v_jacobi_method/common/include/common.hpp"
#include "util/include/util.hpp"

#include <iostream>

namespace romanova_v_jacobi_method {

RomanovaVJacobiMethodMPI::RomanovaVJacobiMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput();
}

bool RomanovaVJacobiMethodMPI::ValidationImpl() {
  bool status = true;

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == 0){
    std::vector<OutType> A = std::get<1>(GetInput());
    OutType x = std::get<0>(GetInput());
    OutType b = std::get<2>(GetInput());
    status = status && !A.empty();
    for(size_t i = 0; i < A.size(); i++) status = status && (A.size() == A[i].size());
    
    status = status && isDiagonallyDominant(A);

    status = status && (A.size() == x.size());
    status = status && (A.size() == b.size());
  }

  MPI_Bcast(&status, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  
  return status;
}

bool RomanovaVJacobiMethodMPI::PreProcessingImpl() {
  
  int rank = 0;
  int n = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);

  std::cout << rank << "\n";

  std::vector<int> send_countsA(n);
  std::vector<int> send_countsb(n);

  std::vector<int> displs_scattA(n);
  std::vector<int> displs_scattb(n);

  if(rank == 0){
    n_ = std::get<1>(GetInput()).size();
  }

  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  x_ = OutType(n_);

  if(rank == 0){
    std::vector<OutType> tempA;
    std::tie(x_, tempA, b_, eps_, maxIterations_) = GetInput();
    //n_ = tempA.size();

    for (const auto &vec : tempA) {
      A_.insert(A_.end(), vec.begin(), vec.end());
    }
    
  }

  MPI_Bcast(&eps_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(maxIterations_&, 1, MPI_INT, 0, MPI_COMM_WORLD);

  size_t delta = n_/n;
  size_t extra = n_%n;

  local_n_ = delta + (rank < extra? 1 : 0);
  st_row_ = rank * delta + (rank < extra? rank:extra);

  if(rank == 0){
    send_countsA = std::vector<int>(n, static_cast<int>(delta * n_));
    send_countsb = std::vector<int>(n, static_cast<int>(delta));
    for(int i = 0; i < extra; i++){
      send_countsA[i]+=n_;
      send_countsb[i]++;
    } 

    for (int i = 1; i < n; i++) {
      displs_scattA[i] = displs_scattA[i - 1] + send_countsA[i-1];
      displs_scattb[i] = displs_scattb[i - 1] + send_countsb[i-1];
    }
  }

  OutType local_data = OutType(local_n_ * n_);
  OutType local_b = OutType(local_n_);

  std::cout << "rank: " << rank << ", x_.size(): " << x_.size() << "\n";

  MPI_Bcast(x_.data(), n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? A_.data() : nullptr, send_countsA.data(), displs_scattA.data(), MPI_DOUBLE,
               local_data.data(), static_cast<int>(local_data.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? b_.data() : nullptr, send_countsb.data(), displs_scattb.data(), MPI_DOUBLE,
               local_b.data(), static_cast<int>(local_b.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  A_ = local_data;
  b_ = local_b;

  std::cout << "hi, my name is " << rank << " and i'm going to show you my data\n";
  std::cout << "my A_:\n";
  for(int i = 0; i < local_n_; i++){
    for(int j = 0; j < n_; j++) std::cout << A_[i*n_+j] << " ";
    std::cout << "\n";
  }
  std::cout << "my b_:\n";
  for(int i = 0; i < local_n_; i++) std::cout << b_[i] << " ";
  std::cout << "\nmy x_:\n";
  for(int i = 0; i < n_; i++) std::cout << x_[i]<<" ";
  std::cout << "\n";

  return true;
}

bool RomanovaVJacobiMethodMPI::RunImpl() {
  return true;
}

bool RomanovaVJacobiMethodMPI::PostProcessingImpl() {
  return true;
}

bool RomanovaVJacobiMethodMPI::isDiagonallyDominant(const std::vector<OutType>& matrix){
  for(int i = 0; i < matrix.size(); i++){
    double sum = 0.0;
    for(int j = 0; j < matrix[i].size(); j++){
      if(i != j) sum += matrix[i][j];
    }
    if(matrix[i][i] <= sum) return false;
  }
  return true;
}

bool RomanovaVJacobiMethodMPI::isConverge(const OutType& prev, const OutType& curr){
  double diff = 0.0;
  for(size_t i = 0; i < prev.size(); i++) diff = std::max(diff, abs(prev[i]-curr[i]));
  return (diff < eps_);
}

}  // romanova_v_jacobi_method
