#include "guseva_a_jarvis/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "guseva_a_jarvis/common/include/common.hpp"

namespace guseva_a_jarvis {

GusevaAJarvisSEQ::GusevaAJarvisSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool GusevaAJarvisSEQ::ValidationImpl() {
  const auto &[width, height, image] = GetInput();
  bool is_size_match = static_cast<int>(image.size()) == width * height;
  bool is_image_binary = std::ranges::all_of(image.begin(), image.end(), [](int x) { return x == 0 || x == 1; });
  bool is_size_possible = width > 0 && height > 0;
  return is_image_binary && is_size_possible && is_size_match;
}

bool GusevaAJarvisSEQ::PreProcessingImpl() {
  const auto &[width, height, image] = GetInput();
  // NOLINTBEGIN
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (image[(y * width) + x] == 1) {
        points_.emplace_back(x, y);
      }
    }
  }
  // NOLINTEND
  return true;
}

size_t GusevaAJarvisSEQ::FindFirst() const {
  size_t start_idx = 0;
  for (size_t i = 0; i < points_.size(); i++) {
    if (points_[i].first < points_[start_idx].first ||
        (points_[i].first == points_[start_idx].first && points_[i].second < points_[start_idx].second)) {
      start_idx = i;
    }
  }
  return start_idx;
}

// NOLINTNEXTLINE
bool GusevaAJarvisSEQ::RunImpl() {
  if (points_.size() == 3) {
    return true;
  }

  size_t start_idx = FindFirst();
  size_t current = start_idx;

  // NOLINTNEXTLINE
  do {
    hull_.push_back(points_[current]);
    size_t next = (current + 1) % points_.size();
    for (size_t i = 0; i < points_.size(); ++i) {
      if (i == current || i == next) {
        continue;
      }
      int x1 = points_[current].first;
      int y1 = points_[current].second;
      int x2 = points_[next].first;
      int y2 = points_[next].second;
      int x3 = points_[i].first;
      int y3 = points_[i].second;

      int cross = ((x2 - x1) * (y3 - y1)) - ((y2 - y1) * (x3 - x1));

      if (cross < 0) {
        next = i;
      } else if (cross == 0) {
        int dist1 = ((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1));
        int dist2 = ((x3 - x1) * (x3 - x1)) + ((y3 - y1) * (y3 - y1));
        if (dist2 > dist1) {
          next = i;
        }
      }
    }

    current = next;

  } while (current != start_idx);

  return true;
}

bool GusevaAJarvisSEQ::PostProcessingImpl() {
  GetOutput().resize(static_cast<int64_t>(std::get<0>(GetInput())) * std::get<1>(GetInput()), 0);
  for (const auto &[x, y] : hull_) {
    GetOutput()[(y * std::get<0>(GetInput())) + x] = 1;
  }
  return true;
}

}  // namespace guseva_a_jarvis
