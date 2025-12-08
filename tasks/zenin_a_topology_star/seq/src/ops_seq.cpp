#include "zenin_a_topology_star/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "zenin_a_topology_star/common/include/common.hpp"

namespace zenin_a_topology_star {

ZeninATopologyStarSEQ::ZeninATopologyStarSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ZeninATopologyStarSEQ::ValidationImpl() {
  return true;
}

bool ZeninATopologyStarSEQ::PreProcessingImpl() {
  return true;
}

bool ZeninATopologyStarSEQ::RunImpl() {
  auto &input = GetInput();
  auto &rows = std::get<0>(input);
  auto &columns = std::get<1>(input);
  auto &matrix = std::get<2>(input);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < columns; ++col) {
      GetOutput()[col] += matrix[(row * columns) + col];
    }
  }
  return true;
}

bool ZeninATopologyStarSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zenin_a_topology_star
