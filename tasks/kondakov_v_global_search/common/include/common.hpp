#pragma once

#include <cstdint>
#include <string>

namespace kondakov_v_global_search {

enum class FunctionType : std::uint8_t { kQuadratic, kSine, kAbs };

struct Params {
  FunctionType func_type = FunctionType::kQuadratic;
  double func_param = 0.0;
  double left = 0.0;
  double right = 0.0;
  double accuracy = 1e-6;
  double reliability = 1.0;
  int max_iterations = 1000;
};

struct Solution {
  double argmin = 0.0;
  double value = 0.0;
  int iterations = 0;
  bool converged = false;
};

using InType = Params;
using OutType = Solution;
using TestType = std::tuple<Params, std::string>;

}  // namespace kondakov_v_global_search
