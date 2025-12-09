#include "telnov_transfer_one_all/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "telnov_transfer_one_all/common/include/common.hpp"
#include "util/include/util.hpp"

namespace telnov_transfer_one_all {

template <typename T>
TelnovTransferOneAllSEQ<T>::TelnovTransferOneAllSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(0);
}

template <typename T>
bool TelnovTransferOneAllSEQ<T>::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput().empty());
}

template <typename T>
bool TelnovTransferOneAllSEQ<T>::PreProcessingImpl() {
  return true;
}

template <typename T>
bool TelnovTransferOneAllSEQ<T>::RunImpl() {
  SetOutput(GetInput());
  return true;
}

template <typename T>
bool TelnovTransferOneAllSEQ<T>::PostProcessingImpl() {
  return !GetOutput().empty();
}

template class TelnovTransferOneAllSEQ<int>;
template class TelnovTransferOneAllSEQ<float>;
template class TelnovTransferOneAllSEQ<double>;

}  // namespace telnov_transfer_one_all
