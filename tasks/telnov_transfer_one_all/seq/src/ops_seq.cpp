#include "telnov_transfer_one_all/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "telnov_transfer_one_all/common/include/common.hpp"
#include "util/include/util.hpp"

namespace telnov_transfer_one_all {

TelnovTransferOneAllSEQ::TelnovTransferOneAllSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(0);
}

bool TelnovTransferOneAllSEQ::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput().empty());
}

bool TelnovTransferOneAllSEQ::PreProcessingImpl() {
  return true;
}

bool TelnovTransferOneAllSEQ::RunImpl() {
  SetOutput(GetInput());
  return true;
}

bool TelnovTransferOneAllSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

template class TelnovTransferOneAllSEQ<int>;
template class TelnovTransferOneAllSEQ<float>;
template class TelnovTransferOneAllSEQ<double>;

}  // namespace telnov_transfer_one_all
