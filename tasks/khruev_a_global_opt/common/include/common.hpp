#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
// Вспомогательная функция для поворота/отражения квадранта
// n: размер текущего квадрата (степень двойки)
// x, y: координаты
// rx, ry: биты, определяющие квадрант
inline void rotate(uint64_t n, uint64_t &x, uint64_t &y, uint64_t rx, uint64_t ry) {
  if (ry == 0) {
    if (rx == 1) {
      x = n - 1 - x;
      y = n - 1 - y;
    }
    // Swap x и y
    uint64_t t = x;
    x = y;
    y = t;
  }
}

// 1. Кривая Гильберта: переводит t [0,1] -> (x, y) [0,1]x[0,1]
// Используем 32-битный порядок кривой (сетка 2^32 x 2^32), что покрывает точность double
inline void d2xy(double t, double &x, double &y) {
  // Ограничиваем t диапазоном [0, 1]
  if (t < 0.0) {
    t = 0.0;
  }
  if (t > 1.0) {
    t = 1.0;
  }

  // Порядок кривой N = 32. Всего точек 2^64 (влезает в uint64_t).
  // Масштабируем t [0, 1] в целое число s [0, 2^64 - 1]
  // Используем (2^64 - 1) как множитель.
  const uint64_t max_dist = static_cast<uint64_t>(-1);  // Все биты 1
  uint64_t s = static_cast<uint64_t>(t * static_cast<double>(max_dist));

  uint64_t ix = 0;
  uint64_t iy = 0;

  // Итеративный алгоритм от младших битов к старшим (bottom-up)
  // n - это размер под-квадрата на текущем уровне
  for (uint64_t n = 1; n < (1ULL << 32); n <<= 1) {
    uint64_t rx = 1 & (s / 2);
    uint64_t ry = 1 & (s ^ rx);

    rotate(n, ix, iy, rx, ry);

    ix += n * rx;
    iy += n * ry;

    s /= 4;
  }

  // Нормализуем обратно в [0, 1]
  // Делим на 2^32 (размер сетки)
  const double scale = 1.0 / 4294967296.0;  // 1.0 / 2^32
  x = static_cast<double>(ix) * scale;
  y = static_cast<double>(iy) * scale;
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
