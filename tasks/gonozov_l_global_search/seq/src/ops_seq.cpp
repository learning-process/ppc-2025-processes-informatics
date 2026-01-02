#include "gonozov_l_global_search/seq/include/ops_seq.hpp"

#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>
#include <algorithm>
#include <limits>

#include "gonozov_l_global_search/common/include/common.hpp"
#include "util/include/util.hpp"

namespace gonozov_l_global_search {
GonozovLGlobalSearchSEQ::GonozovLGlobalSearchSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool GonozovLGlobalSearchSEQ::ValidationImpl() {
  return (std::get<1>(GetInput()) > 1.0) && (std::get<2>(GetInput()) < std::get<3>(GetInput())) && (std::get<4>(GetInput()) > 0);
}

bool GonozovLGlobalSearchSEQ::PreProcessingImpl() {
  return true;
}

namespace {
  double Countingm(double M, double r) {
    return (M == 0.0) ? 1.0 : r * M;
  }

  // double CountingM(int t, double M, std::vector<double> &testSequence, const auto &function )
  // {
  //   if (M == -std::numeric_limits<double>::infinity())
  //   {
  //     M = abs((function(testSequence[1]) - function(testSequence[0])) / (testSequence[1] - testSequence[0]));
  //   }
  //   else
  //   {
  //     double M1 = abs(function(testSequence.back())- function(testSequence[t - 1])) / (testSequence.back() - testSequence[t - 1]);
  //     double M2 = abs(function(testSequence[t]) - function(testSequence.back())) / (testSequence[t] - testSequence.back());
  //     return std::max(M, std::max(M1, M2));
  //   }
  //   //M = -numeric_limits<double>::infinity();
  //   //for (int i = 1; i < testSequence.size(); i++)
  //   //{
  //   //	if (abs((testSequence[i].getZ() - testSequence[i - 1].getZ()) / (testSequence[i] - testSequence[i - 1]).getX()) > M)
  //   //	{
  //   //		M = abs((testSequence[i].getZ() - testSequence[i - 1].getZ()) / (testSequence[i] - testSequence[i - 1]).getX());
  //   //	}
  //   //}
  //   return M;
  // }
  double CountingM(int t, double M,
                   std::vector<double> &testSequence,
                   const auto &function)
  {
    if (M == -std::numeric_limits<double>::infinity())
    {
      return std::abs((function(testSequence[1]) -
                       function(testSequence[0])) /
                      (testSequence[1] - testSequence[0]));
    }
    else
    {
      double M1 = std::abs(function(testSequence.back()) -
                           function(testSequence[t - 1])) /
                  (testSequence.back() - testSequence[t - 1]);

      double M2 = std::abs(function(testSequence[t]) -
                           function(testSequence.back())) /
                  (testSequence[t] - testSequence.back());

      return std::max(M, std::max(M1, M2));
    }
  }

  int Countingt(double m, std::vector<double> &testSequence, const auto &function)
  {
    int t = 1;
    double valueRt = -std::numeric_limits<double>::infinity();
    for (unsigned int i = 1; i < testSequence.size(); i++)
    {
      double subX = (testSequence[i] - testSequence[i - 1]);
      double subZ = (function(testSequence[i]) - function(testSequence[i - 1]));
      double sumZ = (function(testSequence[i]) + function(testSequence[i - 1]));
      double interValueRt = m * subX + subZ * subZ / m / subX - 2 * sumZ;
      if (interValueRt > valueRt)
      {
        valueRt = interValueRt;
        t = i;
      }
    }
    return t;
  }
}

bool GonozovLGlobalSearchSEQ::RunImpl() {
  std::vector<double> testSequence;
  std::function<double(double)> const &function = std::get<0>(GetInput());
  double r = std::get<1>(GetInput());
  double a = std::get<2>(GetInput());
  double b = std::get<3>(GetInput());
  double epsilon = std::get<4>(GetInput());

  testSequence.push_back(a);
  testSequence.push_back(b);

  int t = 1;
  double M = -std::numeric_limits<double>::infinity();

  double global_min_x;
  double global_min_value;
  if (function(a) < function(b))
  {
    global_min_x = a;
    global_min_value = function(a);
  }
  else
  {
    global_min_x = b;
    global_min_value = function(b);
  }

  int iteration_count = 0;
  const int MAX_ITERATIONS = std::numeric_limits<int>::infinity();
  do
  {
    iteration_count++;
    std::sort(testSequence.begin(), testSequence.end());
    M = CountingM(t, M, testSequence, function);

    double m = Countingm(M, r);
    t = Countingt(m, testSequence, function);
    double newElemSequence = 0.5 * (testSequence[t] + testSequence[t - 1]) - 0.5 / m * (function(testSequence[t]) - function(testSequence[t - 1]));
    if (function(newElemSequence) < global_min_value) {
      global_min_x = newElemSequence;
      global_min_value = function(newElemSequence);
    }
    testSequence.push_back(newElemSequence);
    if (iteration_count == MAX_ITERATIONS)  
      break;
  } while (abs(testSequence[t] - testSequence[t - 1]) > epsilon);

  GetOutput() = global_min_x;

  return true;
}

bool GonozovLGlobalSearchSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gonozov_l_global_search
