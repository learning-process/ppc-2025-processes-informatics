#include "romanova_v_jacobi_method_processes/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "romanova_v_jacobi_method_processes/common/include/common.hpp"
#include "util/include/util.hpp"

namespace romanova_v_jacobi_method_processes {

RomanovaVJacobiMethodSEQ::RomanovaVJacobiMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput();
}

bool RomanovaVJacobiMethodSEQ::ValidationImpl() {
  bool status = true;
  //матрица имеет корректные размеры
  std::vector<std::vector<double>> A = std::get<1>(GetInput());
  std::vector<double> x = std::get<0>(GetInput());
  std::vector<double> b = std::get<2>(GetInput());
  status = status && !A.empty();
  for(size_t i = 0; i < A.size(); i++) status = status && (A.size() == A[i].size());
  
  status = status && isDiagonallyDominant(A);

  status = status && (A.size() == x.size());
  status = status && (A.size() == b.size());
  return status;
}

bool RomanovaVJacobiMethodSEQ::PreProcessingImpl() {
  x_ = std::get<0>(GetInput());
  A_ = std::get<1>(GetInput());
  b_ = std::get<2>(GetInput());
  eps_ = std::get<3>(GetInput());
  maxIterations_ = std::get<4>(GetInput());
  size_ = x_.size();
  return true;
}

bool RomanovaVJacobiMethodSEQ::RunImpl() {
  size_t k = 0;
  std::vector<double> prev(x_.size(), 0.0);
  do{
    prev = x_;
    for(size_t i = 0; i < size_; i++){
      double sum = 0.0;
      for(size_t j = 0; j < size_; j++){
        if(i != j){
          sum += A_[i][j]*prev[j];
        }
      }
      x_[i] = (b_[i] - sum)/A_[i][i];
    }
    k++;
  }while(!isConverge(prev,x_) && k <= maxIterations_);

  return true;
}

bool RomanovaVJacobiMethodSEQ::PostProcessingImpl() {
  GetOutput() = x_;
  return true;
}


bool RomanovaVJacobiMethodSEQ::isDiagonallyDominant(const std::vector<std::vector<double>>& matrix){
  for(int i = 0; i < matrix.size(); i++){
    double sum = 0.0;
    for(int j = 0; j < matrix[i].size(); j++){
      if(i != j) sum += matrix[i][j];
    }
    if(matrix[i][i] <= sum) return false;
  }
  return true;
}

bool RomanovaVJacobiMethodSEQ::isConverge(const std::vector<double>& prev, const std::vector<double>& curr){
  double diff = 0.0;
  for(size_t i = 0; i < prev.size(); i++) diff = std::max(diff, abs(prev[i]-curr[i]));
  return (diff < eps_);
}

}  // namespace romanova_v_jacobi_method_processes
