#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"

namespace kosolapov_v_gauss_method_tape_hor_scheme {

struct LinSystem {
  std::vector<std::vector<double>> matrix;
  std::vector<double> r_side;
  LinSystem() = default;
  LinSystem(const LinSystem &other) : matrix(other.matrix), r_side(other.r_side) {};
};

using InType = LinSystem;
using OutType = std::vector<double>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kosolapov_v_gauss_method_tape_hor_scheme
