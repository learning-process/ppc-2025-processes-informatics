#include "levonychev_i_min_val_rows_matrix/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "levonychev_i_min_val_rows_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace levonychev_i_min_val_rows_matrix {

LevonychevIMinValRowsMatrixSEQ::LevonychevIMinValRowsMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  //GetOutput() = 0;
}

bool LevonychevIMinValRowsMatrixSEQ::ValidationImpl() {
  if (GetInput().empty()){
    return false;
  }
  size_t row_length = GetInput()[0].size();
  for (size_t i = 1; i < GetInput().size(); ++i){
    if (GetInput()[i].size() != row_length){
      return false;
    }
  }
  return true;
}

bool LevonychevIMinValRowsMatrixSEQ::PreProcessingImpl() {
  GetOutput().resize(GetInput().size());
  return true;
}

bool LevonychevIMinValRowsMatrixSEQ::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  for (InType i = 0; i < GetInput(); i++) {
    for (InType j = 0; j < GetInput(); j++) {
      for (InType k = 0; k < GetInput(); k++) {
        std::vector<InType> tmp(i + j + k, 1);
        GetOutput() += std::accumulate(tmp.begin(), tmp.end(), 0);
        GetOutput() -= i + j + k;
      }
    }
  }

  const int num_threads = ppc::util::GetNumThreads();
  GetOutput() *= num_threads;

  int counter = 0;
  for (int i = 0; i < num_threads; i++) {
    counter++;
  }

  if (counter != 0) {
    GetOutput() /= counter;
  }
  return GetOutput() > 0;
}

bool LevonychevIMinValRowsMatrixSEQ::PostProcessingImpl() {
  GetOutput() -= GetInput();
  return GetOutput() > 0;
}

}  // namespace levonychev_i_min_val_rows_matrix
