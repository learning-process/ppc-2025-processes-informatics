#include "chyokotov_a_convex_hull_finding/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <queue>
#include <vector>

#include "chyokotov_a_convex_hull_finding/common/include/common.hpp"

namespace chyokotov_a_convex_hull_finding {

ChyokotovConvexHullFindingSEQ::ChyokotovConvexHullFindingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().clear();
  GetInput().reserve(in.size());
  for (const auto &row : in) {
    GetInput().push_back(row);
  }

  GetOutput().clear();
}

bool ChyokotovConvexHullFindingSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return true;
  }

  for (size_t i = 0; i < input.size(); ++i) {
    for (size_t j = 0; j < input[0].size(); ++j) {
      if (input[i][j] != 1 && input[i][j] != 0) {
        return false;
      }
    }
  }

  size_t length_row = input[0].size();
  return std::ranges::all_of(input, [length_row](const auto &row) { return row.size() == length_row; });
}

bool ChyokotovConvexHullFindingSEQ::PreProcessingImpl() {
  return true;
}

std::vector<std::vector<std::pair<int, int>>> ChyokotovConvexHullFindingSEQ::FindComponent() {
  auto picture = GetInput();
  int n = static_cast<int>(picture.size());
  int m = static_cast<int>(picture[0].size());
  std::vector<std::vector<std::pair<int, int>>> components;
  std::vector<std::vector<bool>> was(n, std::vector<bool>(m, false));

  const int directions[4][2] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};

  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < m; ++x) {
      if (picture[y][x] == 1 && !was[y][x]) {
        std::vector<std::pair<int, int>> component;
        std::queue<std::pair<int, int>> q;

        q.push({x, y});
        was[y][x] = true;

        while (!q.empty()) {
          auto [cx, cy] = q.front();
          q.pop();

          component.push_back({cx, cy});

          for (int i = 0; i < 4; ++i) {
            int nx = cx + directions[i][0];
            int ny = cy + directions[i][1];

            if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
              if (picture[ny][nx] == 1 && !was[ny][nx]) {
                was[ny][nx] = true;
                q.push({nx, ny});
              }
            }
          }
        }

        components.push_back(component);
      }
    }
  }

  return components;
}

int ChyokotovConvexHullFindingSEQ::Cross(const std::pair<int, int> &o, const std::pair<int, int> &a,
                                         const std::pair<int, int> &b) {
  return (a.first - o.first) * (b.second - o.second) - (a.second - o.second) * (b.first - o.first);
}

std::vector<std::pair<int, int>> ChyokotovConvexHullFindingSEQ::ConvexHull(std::vector<std::pair<int, int>> x) {
  int n = x.size();

  if (n <= 2) {
    return x;
  }

  std::sort(x.begin(), x.end());

  std::vector<std::pair<int, int>> hull(2 * n);
  int k = 0;

  for (int i = 0; i < n; i++) {
    while (k >= 2 && Cross(hull[k - 2], hull[k - 1], x[i]) <= 0) {
      k--;
    }
    hull[k++] = x[i];
  }

  for (int i = n - 2, t = k + 1; i >= 0; i--) {
    while (k >= t && Cross(hull[k - 2], hull[k - 1], x[i]) <= 0) {
      k--;
    }
    hull[k++] = x[i];
  }

  hull.resize(k - 1);
  return hull;
}

bool ChyokotovConvexHullFindingSEQ::RunImpl() {
  auto &picture = GetInput();
  if (picture.empty()) {
    return true;
  }
  auto &output = GetOutput();
  auto components = FindComponent();
  for (auto &i : components) {
    output.push_back(ConvexHull(i));
  }

  return true;
}

bool ChyokotovConvexHullFindingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace chyokotov_a_convex_hull_finding
