#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

#include "task/include/task.hpp"

namespace khruev_a_global_opt {

struct SearchData {
  int func_id;
  double ax, bx;
  double ay, by;
  double epsilon;
  int max_iter;
  double r;
};

struct SearchResult {
  double x;
  double y;
  double value;
  int iter_count;
};

using InType = SearchData;
using OutType = SearchResult;
using BaseTask = ppc::task::Task<InType, OutType>;
using TestType = std::tuple<std::string, int, double, double, double, double, double>;

// 1. Кривая Гильберта: переводит t [0,1] -> (x, y) [0,1]x[0,1]
inline void d2xy(double t, double &x, double &y) {
  const int n = 1 << 16;  //
  int curr_t = static_cast<int>(t * (static_cast<double>(n) * n - 1));

  int rx, ry, s;
  int ix = 0;
  int iy = 0;
  for (s = 1; s < n; s *= 2) {
    rx = 1 & (curr_t / 2);
    ry = 1 & (curr_t ^ rx);
    // Rotate/flip
    if (ry == 0) {
      if (rx == 1) {
        ix = s - 1 - ix;
        iy = s - 1 - iy;
      }
      std::swap(ix, iy);
    }
    ix += s * rx;
    iy += s * ry;
    curr_t /= 4;
  }
  x = static_cast<double>(ix) / n;
  y = static_cast<double>(iy) / n;
}

inline double target_function(int id, double x, double y) {
  // x, y приходят из [0, 1]x[0, 1], нужно масштабировать внутри если надо
  if (id == 1) {
    // Простая яма (x-0.5)^2 + (y-0.5)^2
    return (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
  }
  if (id == 2) {
    // Rastrigin-like (многоэкстремальная)
    // Масштабируем [0,1] -> [-2, 2] для наглядности
    double sx = (x - 0.5) * 4.0;
    double sy = (y - 0.5) * 4.0;
    return (sx * sx - 10 * std::cos(2 * 3.14159 * sx)) + (sy * sy - 10 * std::cos(2 * 3.14159 * sy)) + 20;
  }
  return 0.0;
}

// Структура для хранения испытаний
struct Trial {
  double x;  // координата на отрезке [0, 1]
  double z;  // значение функции
  int id;    // исходный индекс

  bool operator<(const Trial &other) const {
    return x < other.x;
  }
};

}  // namespace khruev_a_global_opt
