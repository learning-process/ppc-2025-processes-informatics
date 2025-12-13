#include "telnov_strongin_algorithm/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "telnov_strongin_algorithm/common/include/common.hpp"
#include "util/include/util.hpp"

namespace telnov_strongin_algorithm {

TelnovStronginAlgorithmSEQ::TelnovStronginAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TelnovStronginAlgorithmSEQ::ValidationImpl() {
  const auto &in = GetInput();
  return in.eps > 0 && in.b > in.a;
}

bool TelnovStronginAlgorithmSEQ::PreProcessingImpl() {
  return true;
}

bool TelnovStronginAlgorithmSEQ::RunImpl() {
  const auto &in = GetInput();
  const double a = in.a;
  const double b = in.b;
  const double eps = in.eps;

  auto f = [](double x) { return (x - 1) * (x - 1) + 1; };  // тестовая функция

  std::vector<double> X = {a, b};
  std::vector<double> F = {f(a), f(b)};

  while ((X.back() - X.front()) > eps) {
    double M = 0.0;
    for (size_t i = 1; i < X.size(); ++i) {
      M = std::max(M, std::abs(F[i] - F[i - 1]) / (X[i] - X[i - 1]));
    }
    if (M == 0) {
      M = 1.0;
    }

    double r = 2.0;
    double maxR = -1e9;
    size_t t = 0;

    for (size_t i = 1; i < X.size(); ++i) {
      double Ri = r * (X[i] - X[i - 1]) + (F[i] - F[i - 1]) * (F[i] - F[i - 1]) / (r * (X[i] - X[i - 1])) -
                  2 * (F[i] + F[i - 1]);
      if (Ri > maxR) {
        maxR = Ri;
        t = i;
      }
    }

    double x_new = 0.5 * (X[t] + X[t - 1]) - (F[t] - F[t - 1]) / (2 * M);
    X.insert(X.begin() + t, x_new);
    F.insert(F.begin() + t, f(x_new));
  }

  GetOutput() = *std::min_element(F.begin(), F.end());
  return true;
}

bool TelnovStronginAlgorithmSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace telnov_strongin_algorithm
