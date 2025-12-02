#include "romanova_v_jacobi_method_processes/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "romanova_v_jacobi_method_processes/common/include/common.hpp"
#include "util/include/util.hpp"

namespace romanova_v_jacobi_method_processes {

RomanovaVJacobiMethodSEQ::RomanovaVJacobiMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool RomanovaVJacobiMethodSEQ::PreProcessingImpl() {
  x = std::get<0>(GetInput());
  A = std::get<1>(GetInput());
  b = std::get<2>(GetInput());
  eps = std::get<3>(GetInput());
  maxIterations = std::get<4>(GetInput());
  size = x.size();
  return true;
}

bool RomanovaVJacobiMethodSEQ::ValidationImpl() {
  bool status = true;
  //матрица имеет корректные размеры
  status &&= !A.empty();
  for(size_t i = 0; i < A.size(); i++) status &&= (A.size() == A[i].size());
  
  status &&= isDiagonallyDominant(A);

  status &&= (A.size() == x.size());
  status &&= (A.size() == b.size());
  return status;
}

bool RomanovaVJacobiMethodSEQ::RunImpl() {
  size_t k = 0;
  std::vector<double> prev(x.size(), 0.0);
  do{
    prev = x;
    for(size_t i = 0; i < size; i++){
      double sum = 0.0;
      for(size_t j = 0; j < size; j++){
        if(i != j){
          sum += A[i][j]*prev[j];
        }
      }
      x[i] = (b[i] - sum)/A[i][j];
    }
    k++;
  }while(!isConverge(prev,x) && k <= maxIterations);

  return true;
}

bool RomanovaVJacobiMethodSEQ::PostProcessingImpl() {
  GetOutput() = x;
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
  return (diff < eps);
}

}  // namespace romanova_v_jacobi_method_processes
