#include "luzan_e_simps_int/seq/include/ops_seq.hpp"

#include <cstddef>
#include <tuple>
#include <vector>

#include "luzan_e_simps_int/common/include/common.hpp"

namespace luzan_e_simps_int {

LuzanESimpsIntSEQ::LuzanESimpsIntSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool LuzanESimpsIntSEQ::ValidationImpl() {
  int n = std::get<0>(GetInput());
  double a = std::get<0>(std::get<1>(GetInput()));
  double b = std::get<1>(std::get<1>(GetInput()));
  double c = std::get<0>(std::get<2>(GetInput()));
  double d = std::get<1>(std::get<2>(GetInput()));

  bool flag = (a < b) && (c < d) && (n % 2 == 0) && (n > 0);
  return flag;
}

bool LuzanESimpsIntSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool LuzanESimpsIntSEQ::RunImpl() {
  double a = 0.0, b = 0.0;
  double c = 0.0, d = 0.0;
  int n = 0;  // кол-во отрезков

  // getting data
  n = std::get<0>(GetInput());
  a = std::get<0>(std::get<1>(GetInput()));
  b = std::get<1>(std::get<1>(GetInput()));
  c = std::get<0>(std::get<2>(GetInput()));
  d = std::get<1>(std::get<2>(GetInput()));

  double hx = (b - a) / n;
  double hy = (d - c) / n;

  double sum = 0;
  double wx = 1.0, wy = 1.0;
  double x = 0.0, y = 0.0;

  for (int i = 0; i <= n; i++) {
    x = a + hx * i;
    wx = 1.0;
    if (i != 0 && i != n) {
      if (i % 2 == 1) {
        wx = 4.0;
      } else {
        wx = 2.0;
      }
    }

    for (int j = 0; j <= n; j++) {
      wy = 1.0;
      y = c + hy * j;

      if (j != 0 && j != n) {
        if (j % 2 == 1) {
          wy = 4.0;
        } else {
          wy = 2.0;
        }
      }
      sum += wy * wx * f(x, y);
    }
  }
  sum = sum * hx * hy / (3 * 3);
  GetOutput() = sum;
  return true;
}

bool LuzanESimpsIntSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace luzan_e_simps_int
