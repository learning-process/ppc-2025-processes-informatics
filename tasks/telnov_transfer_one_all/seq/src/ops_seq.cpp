#include "telnov_transfer_one_all/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "telnov_transfer_one_all/common/include/common.hpp"
#include "util/include/util.hpp"

namespace telnov_transfer_one_all {

TelnovTransferOneAllSEQ::TelnovTransferOneAllSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TelnovTransferOneAllSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool TelnovTransferOneAllSEQ::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

bool TelnovTransferOneAllSEQ::RunImpl() {
  SetOutput(GetInput());
  return true;
}

bool TelnovTransferOneAllSEQ::PostProcessingImpl() {
    return !GetOutput().empty();
}

}  // namespace telnov_transfer_one_all
