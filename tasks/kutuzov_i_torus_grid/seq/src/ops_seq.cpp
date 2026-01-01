#include "kutuzov_i_torus_grid/seq/include/ops_seq.hpp"

#include <tuple>
#include <vector>

#include "kutuzov_i_torus_grid/common/include/common.hpp"

namespace kutuzov_i_torus_grid {

KutuzovIThorusGridSEQ::KutuzovIThorusGridSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(std::vector<int>{}, "");
}

bool KutuzovIThorusGridSEQ::ValidationImpl() {
  return true;
}

bool KutuzovIThorusGridSEQ::PreProcessingImpl() {
  return true;
}

bool KutuzovIThorusGridSEQ::RunImpl() {
  std::vector<int> route = {1, 2, 3, 4, 5};
  std::string message = "msg";
  GetOutput() = std::make_tuple(route, message);

  return true;
}

bool KutuzovIThorusGridSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kutuzov_i_torus_grid
