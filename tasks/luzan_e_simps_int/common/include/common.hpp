#pragma once

#include <cmath>
#include <tuple>

#include "task/include/task.hpp"

namespace luzan_e_simps_int {

// n, a-b, c-d, func_num
using InType = std::tuple<int, std::tuple<int, int>, std::tuple<int, int>, int>;
using OutType = double;
// n, a-b, c-d, func_num
using TestType = std::tuple<int, std::tuple<int, int>, std::tuple<int, int>, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

const double kEpsilon = 0.0001;

inline double F1(double x, double y) {
  return (pow(x, 5) / 5.0) + (y * sin(y)) + 2.0;
}

inline double F2(double x, double y) {
  return (pow(x, 5) / 5.0) + (y * cos(y)) + 2.0;
}


inline double GetWeight(int i, int n) {
  if (i == 0 || i == n) 
    return 1.0;
  if (i % 2 == 1) {
    return 4.0;
  } 
  return 2.0;
} 

inline auto GetFunc(int num) {
  switch(num) {
  case 1:
    return &F1;
  case 2:
    return &F2;
  }
  return &F1;
}

}  // namespace luzan_e_simps_int
