#include "sizov_d_global_search/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>  // DEBUG
#include <limits>

#include "sizov_d_global_search/common/include/common.hpp"

namespace detail_seq {

static double EstimateM(const std::vector<double> &points, const std::vector<double> &values, double reliability) {
  double max_slope = 0.0;
  for (std::size_t i = 1; i < points.size(); ++i) {
    const double dx = points[i] - points[i - 1];
    if (dx <= 0.0) {
      continue;
    }
    const double slope = std::abs(values[i] - values[i - 1]) / dx;
    max_slope = std::max(max_slope, slope);
  }

  if (max_slope == 0.0) {
    return 1.0;
  }
  return reliability * max_slope;
}

static double CalcCharacteristic(const std::vector<double> &points, const std::vector<double> &values,
                                 std::size_t right_idx, double m) {
  const double dx = points[right_idx] - points[right_idx - 1];
  if (dx <= std::numeric_limits<double>::epsilon()) {
    return -std::numeric_limits<double>::infinity();
  }
  const double df = values[right_idx] - values[right_idx - 1];
  return m * dx + (df * df) / (m * dx) - 2.0 * (values[right_idx] + values[right_idx - 1]);
}

static double CalcNewPoint(const std::vector<double> &points, const std::vector<double> &values, std::size_t right_idx,
                           double m) {
  const double df = values[right_idx] - values[right_idx - 1];
  return 0.5 * (points[right_idx] + points[right_idx - 1]) - df / (2.0 * m);
}

}  // namespace detail_seq

namespace sizov_d_global_search {

SizovDGlobalSearchSEQ::SizovDGlobalSearchSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool SizovDGlobalSearchSEQ::ValidationImpl() {
  const auto &problem = GetInput();
  return problem.func && problem.left < problem.right && problem.accuracy > 0.0 && problem.reliability > 0.0 &&
         problem.max_iterations > 0;
}

bool SizovDGlobalSearchSEQ::PreProcessingImpl() {
  const auto &problem = GetInput();

  std::cout << "[DEBUG][ops_seq.cpp] START: left=" << problem.left << " right=" << problem.right
            << " accuracy=" << problem.accuracy << " reliability=" << problem.reliability
            << " max_iter=" << problem.max_iterations << "\n";

  points_.clear();
  values_.clear();
  iterations_ = 0;
  converged_ = false;

  const double left_value = problem.func(problem.left);
  const double right_value = problem.func(problem.right);

  points_.push_back(problem.left);
  points_.push_back(problem.right);
  values_.push_back(left_value);
  values_.push_back(right_value);

  std::cout << "[DEBUG][ops_seq.cpp] Initial points: xL=" << problem.left << " f=" << left_value
            << "; xR=" << problem.right << " f=" << right_value << "\n";

  if (left_value <= right_value) {
    best_point_ = problem.left;
    best_value_ = left_value;
  } else {
    best_point_ = problem.right;
    best_value_ = right_value;
  }

  std::cout << "[DEBUG][ops_seq.cpp] Initial best: x=" << best_point_ << " f=" << best_value_ << "\n";

  GetOutput() = {best_point_, best_value_, 0, false};
  return true;
}

bool SizovDGlobalSearchSEQ::RunImpl() {
  const auto &problem = GetInput();

  int performed_iterations = 0;

  for (int iter = 0; iter < problem.max_iterations; ++iter) {
    performed_iterations = iter + 1;

    // 1) Оценка константы M
    const double m = detail_seq::EstimateM(points_, values_, problem.reliability);

    // -----------------------------------------------------------
    // DEBUG (каждые 50 итераций)
    // -----------------------------------------------------------
    if (iter % 50 == 0) {
      std::cout << "[DEBUG][ops_seq.cpp][iter=" << iter << "] " << "M=" << m << " points=" << points_.size() << "\n";
    }

    // 2) Поиск интервала с максимальной характеристикой
    double best_characteristic = -std::numeric_limits<double>::infinity();
    std::size_t best_right_idx = 1;

    for (std::size_t idx = 1; idx < points_.size(); ++idx) {
      const double characteristic = detail_seq::CalcCharacteristic(points_, values_, idx, m);
      if (characteristic > best_characteristic) {
        best_characteristic = characteristic;
        best_right_idx = idx;
      }
    }

    // DEBUG (каждые 50 итераций)
    if (iter % 50 == 0) {
      std::cout << "[DEBUG][ops_seq.cpp][iter=" << iter << "] " << "Best interval: (" << points_[best_right_idx - 1]
                << ", " << points_[best_right_idx] << ") idx=" << best_right_idx
                << " characteristic=" << best_characteristic << "\n";
    }

    // 3) Проверка критерия остановки
    const double interval = points_[best_right_idx] - points_[best_right_idx - 1];
    if (interval <= problem.accuracy) {
      if (iter % 50 == 0) {
        std::cout << "[DEBUG][ops_seq.cpp][iter=" << iter << "] STOP: interval <= accuracy\n";
      }

      converged_ = true;
      break;
    }

    // 4) Новая точка
    const double new_point = detail_seq::CalcNewPoint(points_, values_, best_right_idx, m);
    const double new_value = problem.func(new_point);

    // DEBUG (каждые 50 итераций)
    if (iter % 50 == 0) {
      std::cout << "[DEBUG][ops_seq.cpp][iter=" << iter << "] New point: x=" << new_point << " f(x)=" << new_value
                << "\n";
    }

    // 5) Вставка точки
    const auto insert_pos = static_cast<std::ptrdiff_t>(best_right_idx);
    points_.insert(points_.begin() + insert_pos, new_point);
    values_.insert(values_.begin() + insert_pos, new_value);

    // 6) Обновление минимума
    if (new_value < best_value_) {
      if (iter % 50 == 0) {
        std::cout << "[DEBUG][ops_seq.cpp][iter=" << iter << "] New BEST: x=" << new_point << " f=" << new_value
                  << " (prev " << best_value_ << ")\n";
      }

      best_point_ = new_point;
      best_value_ = new_value;
    }
  }

  iterations_ = performed_iterations;

  // -----------------------------------------------------------
  // Итоговое DEBUG-сообщение
  // -----------------------------------------------------------
  std::cout << "[DEBUG][ops_seq.cpp] FINISH: converged=" << converged_ << " best_x=" << best_point_
            << " best_f=" << best_value_ << " iterations=" << iterations_ << "\n";

  GetOutput() = {best_point_, best_value_, iterations_, converged_};
  return true;
}

bool SizovDGlobalSearchSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_global_search
