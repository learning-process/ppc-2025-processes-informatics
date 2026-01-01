#include "chyokotov_a_convex_hull_finding/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <queue>
#include <vector>

#include "chyokotov_a_convex_hull_finding/common/include/common.hpp"

namespace chyokotov_a_convex_hull_finding {

ChyokotovConvexHullFindingMPI::ChyokotovConvexHullFindingMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().clear();
  GetInput().reserve(in.size());
  for (const auto &row : in) {
    GetInput().push_back(row);
  }

  GetOutput().clear();
}

bool ChyokotovConvexHullFindingMPI::ValidationImpl() {
  int rank{};
  int is_valid = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &input = GetInput();
    if (!input.empty()) {
      for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input[0].size(); ++j) {
          if ((input[i][j] != 1 && input[i][j] != 0) || (input[0].size() != input[i].size())) {
            is_valid = 0;
            break;
          }
        }
      }
    }
  }
  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return (is_valid == 1);
}

bool ChyokotovConvexHullFindingMPI::PreProcessingImpl() {
  return true;
}

std::pair<std::vector<int>, int> ChyokotovConvexHullFindingMPI::DistributeImageData(int rank, int size, int width,
                                                                                    int height) {
  const auto &input = GetInput();
  auto [start_row, end_row] = CalculateRowDistribution(rank, size, height);
  int local_rows = end_row - start_row;

  std::vector<int> local_data(local_rows * width);

  if (rank == 0) {
    std::vector<int> send_counts(size);
    std::vector<int> displs(size);

    for (int i = 0; i < size; i++) {
      auto [proc_start, proc_end] = CalculateRowDistribution(i, size, height);
      send_counts[i] = (proc_end - proc_start) * width;
      displs[i] = proc_start * width;
    }

    std::vector<int> flat_input;
    for (const auto &row : input) {
      flat_input.insert(flat_input.end(), row.begin(), row.end());
    }

    MPI_Scatterv(flat_input.data(), send_counts.data(), displs.data(), MPI_INT, local_data.data(), local_rows * width,
                 MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, local_data.data(), local_rows * width, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return {local_data, local_rows};
}

std::pair<int, int> ChyokotovConvexHullFindingMPI::CalculateRowDistribution(int rank, int size, int height) {
  int base_rows = height / size;
  int remainder = height % size;

  int start_row = (rank * base_rows) + std::min(rank, remainder);
  int end_row = start_row + base_rows + (rank < remainder ? 1 : 0);

  return {start_row, end_row};
}

std::vector<std::vector<std::pair<int, int>>> ChyokotovConvexHullFindingMPI::FindConnectedComponentsMPI(
    int rank, int size, int start_row, int end_row, int width, int height, const std::vector<int> &local_data) {
  int local_rows = end_row - start_row;
  if (local_rows == 0) {
    return {};
  }

  bool has_top = start_row > 0;
  bool has_bottom = end_row < height;
  int extended_rows = local_rows + has_top + has_bottom;

  std::vector<int> extended(extended_rows * width, 0);

  int offset = has_top ? 1 : 0;
  for (int i = 0; i < local_rows; i++) {
    int src_idx = i * width;
    int dst_idx = (offset + i) * width;
    std::copy_n(&local_data[src_idx], width, &extended[dst_idx]);
  }

  ExchangeBoundaryRows(has_top, has_bottom, rank, size, width, local_data, extended);

  int global_y_offset = start_row - has_top;
  auto all = ProcessExtendedRegion(extended, extended_rows, width, global_y_offset);

  return FilterLocalComponents(all, start_row, end_row);
}

std::vector<std::pair<int, int>> ChyokotovConvexHullFindingMPI::ConvexHullAndrew(
    const std::vector<std::pair<int, int>> &points) {
  if (points.size() <= 1) {
    return points;
  }

  std::vector<std::pair<int, int>> sorted_points = points;
  std::sort(sorted_points.begin(), sorted_points.end());

  auto last = std::unique(sorted_points.begin(), sorted_points.end());
  sorted_points.erase(last, sorted_points.end());

  if (sorted_points.size() <= 2) {
    return sorted_points;
  }

  std::vector<std::pair<int, int>> lower_hull;
  std::vector<std::pair<int, int>> upper_hull;

  for (const auto &p : sorted_points) {
    while (lower_hull.size() >= 2) {
      const auto &a = lower_hull[lower_hull.size() - 2];
      const auto &b = lower_hull[lower_hull.size() - 1];

      long long cross = static_cast<long long>(b.first - a.first) * (p.second - a.second) -
                        static_cast<long long>(b.second - a.second) * (p.first - a.first);

      if (cross >= 0) {
        lower_hull.pop_back();
      } else {
        break;
      }
    }
    lower_hull.push_back(p);
  }

  for (auto it = sorted_points.rbegin(); it != sorted_points.rend(); ++it) {
    while (upper_hull.size() >= 2) {
      const auto &a = upper_hull[upper_hull.size() - 2];
      const auto &b = upper_hull[upper_hull.size() - 1];

      long long cross = static_cast<long long>(b.first - a.first) * (it->second - a.second) -
                        static_cast<long long>(b.second - a.second) * (it->first - a.first);

      if (cross >= 0) {
        upper_hull.pop_back();
      } else {
        break;
      }
    }
    upper_hull.push_back(*it);
  }

  std::vector<std::pair<int, int>> hull;
  hull.reserve(lower_hull.size() + upper_hull.size() - 2);

  for (size_t i = 0; i < lower_hull.size() - 1; ++i) {
    hull.push_back(lower_hull[i]);
  }

  for (size_t i = 0; i < upper_hull.size() - 1; ++i) {
    hull.push_back(upper_hull[i]);
  }

  return hull;
}

void ChyokotovConvexHullFindingMPI::ExchangeBoundaryRows(bool top, bool bottom, int rank, int size, int width,
                                                         const std::vector<int> &local, std::vector<int> &ext) {
  if (top) {
    std::vector<int> recv(width);
    MPI_Sendrecv(local.data(), width, MPI_INT, rank - 1, 0, recv.data(), width, MPI_INT, rank - 1, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    std::copy(recv.begin(), recv.end(), ext.begin());
  }

  if (bottom && rank + 1 < size) {
    std::vector<int> recv(width);
    MPI_Sendrecv(&local[local.size() - width], width, MPI_INT, rank + 1, 1, recv.data(), width, MPI_INT, rank + 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::copy(recv.begin(), recv.end(), ext.end() - width);
  }
}

std::vector<std::vector<std::pair<int, int>>> ChyokotovConvexHullFindingMPI::ProcessExtendedRegion(
    const std::vector<int> &extended_pixels, int extended_rows, int width, int global_y_offset) {
  std::vector<std::vector<bool>> visited(extended_rows, std::vector<bool>(width, false));
  std::vector<std::vector<std::pair<int, int>>> all_components;

  for (int y = 0; y < extended_rows; ++y) {
    for (int x = 0; x < width; ++x) {
      size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);

      if (extended_pixels[idx] == 1 && !visited[y][x]) {
        auto component = ExtractComponent(x, y, extended_pixels, visited, width, extended_rows, global_y_offset);
        all_components.push_back(std::move(component));
      }
    }
  }

  return all_components;
}

std::vector<std::pair<int, int>> ChyokotovConvexHullFindingMPI::ExtractComponent(
    int start_x, int start_y, const std::vector<int> &extended_pixels, std::vector<std::vector<bool>> &visited,
    int width, int extended_rows, int global_y_offset) {
  std::vector<std::pair<int, int>> component;
  std::queue<std::pair<int, int>> queue;
  queue.emplace(start_x, start_y);
  visited[start_y][start_x] = true;

  const std::array<std::pair<int, int>, 4> directions = {{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}};

  while (!queue.empty()) {
    auto [x, y] = queue.front();
    queue.pop();

    component.emplace_back(x, global_y_offset + y);

    for (const auto &[dx, dy] : directions) {
      int nx = x + dx;
      int ny = y + dy;

      if (nx >= 0 && nx < width && ny >= 0 && ny < extended_rows) {
        size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(width) + static_cast<size_t>(nx);

        if (extended_pixels[nidx] == 1 && !visited[ny][nx]) {
          visited[ny][nx] = true;
          queue.emplace(nx, ny);
        }
      }
    }
  }

  return component;
}

std::vector<std::vector<std::pair<int, int>>> ChyokotovConvexHullFindingMPI::FilterLocalComponents(
    const std::vector<std::vector<std::pair<int, int>>> &all_components, int start_row, int end_row) {
  std::vector<std::vector<std::pair<int, int>>> local_components;

  for (const auto &component : all_components) {
    if (component.empty()) {
      continue;
    }

    int min_y = component[0].second;
    for (const auto &[x, y] : component) {
      if (y < min_y) {
        min_y = y;
      }
    }

    if (min_y >= start_row && min_y < end_row) {
      local_components.push_back(component);
    }
  }

  return local_components;
}

void ChyokotovConvexHullFindingMPI::SendHullsToRank0(const std::vector<std::vector<std::pair<int, int>>> &local_hulls) {
  int hull_count = static_cast<int>(local_hulls.size());
  MPI_Send(&hull_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

  if (hull_count == 0) {
    return;
  }

  std::vector<int> hull_sizes;
  std::vector<int> flat_points;

  for (const auto &hull : local_hulls) {
    int size = static_cast<int>(hull.size());
    hull_sizes.push_back(size);

    for (const auto &[x, y] : hull) {
      flat_points.push_back(x);
      flat_points.push_back(y);
    }
  }

  MPI_Send(hull_sizes.data(), hull_count, MPI_INT, 0, 1, MPI_COMM_WORLD);

  int total_points = static_cast<int>(flat_points.size() / 2);
  MPI_Send(&total_points, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);

  if (total_points > 0) {
    MPI_Send(flat_points.data(), static_cast<int>(flat_points.size()), MPI_INT, 0, 3, MPI_COMM_WORLD);
  }
}

void ChyokotovConvexHullFindingMPI::ReceiveHullsFromRank(int src_rank, std::vector<int> &all_sizes,
                                                         std::vector<int> &global_flat) {
  int hull_count = 0;
  MPI_Recv(&hull_count, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (hull_count == 0) {
    return;
  }

  std::vector<int> sizes(static_cast<size_t>(hull_count));
  MPI_Recv(sizes.data(), hull_count, MPI_INT, src_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  all_sizes.insert(all_sizes.end(), sizes.begin(), sizes.end());

  int total_points = 0;
  MPI_Recv(&total_points, 1, MPI_INT, src_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (total_points > 0) {
    std::vector<int> points_data(static_cast<size_t>(total_points) * 2);
    MPI_Recv(points_data.data(), total_points * 2, MPI_INT, src_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    global_flat.insert(global_flat.end(), points_data.begin(), points_data.end());
  }
}

std::vector<std::vector<std::pair<int, int>>> ChyokotovConvexHullFindingMPI::GatherHullsOnRank0(
    int rank, int size, const std::vector<std::vector<std::pair<int, int>>> &local_hulls) {
  if (rank == 0) {
    std::vector<int> all_sizes;
    std::vector<int> global_flat;

    for (const auto &hull : local_hulls) {
      all_sizes.push_back(static_cast<int>(hull.size()));
      for (const auto &[x, y] : hull) {
        global_flat.push_back(x);
        global_flat.push_back(y);
      }
    }

    for (int src = 1; src < size; ++src) {
      ReceiveHullsFromRank(src, all_sizes, global_flat);
    }

    std::vector<std::vector<std::pair<int, int>>> all_hulls;
    size_t idx = 0;

    for (int hull_size : all_sizes) {
      std::vector<std::pair<int, int>> hull;
      for (int i = 0; i < hull_size; ++i, idx += 2) {
        hull.emplace_back(global_flat[idx], global_flat[idx + 1]);
      }
      all_hulls.push_back(std::move(hull));
    }

    return all_hulls;
  } else {
    SendHullsToRank0(local_hulls);
    return {};
  }
}

std::vector<std::vector<std::pair<int, int>>> ChyokotovConvexHullFindingMPI::BroadcastResultToAllRanks(
    int rank, const std::vector<std::vector<std::pair<int, int>>> &global_hulls_on_rank0) {
  int total_hulls = 0;

  if (rank == 0) {
    total_hulls = static_cast<int>(global_hulls_on_rank0.size());
  }

  MPI_Bcast(&total_hulls, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_hulls == 0) {
    return {};
  }

  std::vector<int> all_sizes(total_hulls);
  if (rank == 0) {
    for (int i = 0; i < total_hulls; ++i) {
      all_sizes[i] = static_cast<int>(global_hulls_on_rank0[i].size());
    }
  }

  MPI_Bcast(all_sizes.data(), total_hulls, MPI_INT, 0, MPI_COMM_WORLD);

  int total_points = 0;
  if (rank == 0) {
    for (const auto &hull : global_hulls_on_rank0) {
      total_points += static_cast<int>(hull.size());
    }
  }

  MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> flat_points(static_cast<size_t>(total_points) * 2);
  if (rank == 0) {
    size_t idx = 0;
    for (const auto &hull : global_hulls_on_rank0) {
      for (const auto &[x, y] : hull) {
        flat_points[idx++] = x;
        flat_points[idx++] = y;
      }
    }
  }

  MPI_Bcast(flat_points.data(), total_points * 2, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<std::vector<std::pair<int, int>>> result;
  size_t idx = 0;

  for (int i = 0; i < total_hulls; ++i) {
    std::vector<std::pair<int, int>> hull;
    int hull_size = all_sizes[i];
    hull.reserve(static_cast<size_t>(hull_size));

    for (int j = 0; j < hull_size; ++j, idx += 2) {
      hull.emplace_back(flat_points[idx], flat_points[idx + 1]);
    }

    result.push_back(std::move(hull));
  }

  return result;
}
bool ChyokotovConvexHullFindingMPI::RunImpl() {
  int rank{};
  int size{};

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n{};
  int m{};

  if (rank == 0) {
    n = static_cast<int>(GetInput().size());
    if (n != 0) {
      m = static_cast<int>(GetInput()[0].size());
    }
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (n == 0) {
    return true;
  }
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

  auto [local_data, local_rows] = DistributeImageData(rank, size, m, n);

  auto [start_row, end_row] = CalculateRowDistribution(rank, size, n);

  auto components = FindConnectedComponentsMPI(rank, size, start_row, end_row, m, n, local_data);

  std::vector<std::vector<std::pair<int, int>>> convex_hulls;
  convex_hulls.reserve(components.size());
  for (const auto &component : components) {
    if (component.size() >= 3) {
      convex_hulls.push_back(ConvexHullAndrew(component));
    } else if (!component.empty()) {
      convex_hulls.push_back(component);
    }
  }

  auto global_hulls = GatherHullsOnRank0(rank, size, convex_hulls);

  GetOutput() = BroadcastResultToAllRanks(rank, global_hulls);

  return true;
}

bool ChyokotovConvexHullFindingMPI::PostProcessingImpl() {
  return true;
}

}  // namespace chyokotov_a_convex_hull_finding
