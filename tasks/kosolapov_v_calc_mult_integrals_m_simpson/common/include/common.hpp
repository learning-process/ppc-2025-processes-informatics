#pragma once

#include <cmath>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace kosolapov_v_calc_mult_integrals_m_simpson {

constexpr double kPi = 3.14159265358979323846;
using InType = std::tuple<int, int>;
using OutType = double;
using TestType = std::tuple<int, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kosolapov_v_calc_mult_integrals_m_simpson
