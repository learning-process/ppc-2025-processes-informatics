#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>
#include <tuple>

#include "task/include/task.hpp"

namespace romanov_a_crs_product {

constexpr double kEps = 1e-6;

class CRS{
private:
    size_t n;
    
    std::vector<double> value;
    std::vector<size_t> column;
    std::vector<size_t> row_index;

public:

  CRS(size_t n): n(n) {
    row_index.resize(n + 1, 0);
  }

  size_t size() const {
    return n;
  }

  size_t getNonZeroCount() const {
    return value.size();
  }

  bool operator==(const CRS& other) const {
    if (n != other.n || column != other.column || row_index != other.row_index) {
      return false;
    }
    for (size_t i = 0; i < value.size(); ++i) {
      if (std::abs(value[i] - other.value[i]) > kEps) {
          return false;
      }
    }
    return true;
  }
};

using InType = std::tuple<size_t, CRS, CRS>;
using OutType = CRS;
using TestType = std::tuple<size_t, CRS, CRS, CRS>;
;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace romanov_a_crs_product
