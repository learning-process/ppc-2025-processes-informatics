#include "baldin_a_gauss_filter/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "baldin_a_gauss_filter/common/include/common.hpp"
#include "util/include/util.hpp"

namespace baldin_a_gauss_filter {

BaldinAGaussFilterSEQ::BaldinAGaussFilterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BaldinAGaussFilterSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool BaldinAGaussFilterSEQ::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

bool BaldinAGaussFilterSEQ::RunImpl() {
  if (GetInput() == 0) {
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

bool BaldinAGaussFilterSEQ::PostProcessingImpl() {
  GetOutput() -= GetInput();
  return GetOutput() > 0;
}

}  // namespace baldin_a_gauss_filter
