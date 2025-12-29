#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yurkin_g_shellbetcher/common/include/common.hpp"
#include "yurkin_g_shellbetcher/mpi/include/ops_mpi.hpp"
#include "yurkin_g_shellbetcher/seq/include/ops_seq.hpp"

namespace yurkin_g_shellbetcher {

using TestVec = std::vector<int>;

static long long ComputeExpectedChecksumSeq(int n) {
  std::vector<int> data;
  data.reserve(static_cast<size_t>(n));
  std::mt19937 rng(static_cast<unsigned int>(n));
  std::uniform_int_distribution<int> dist(0, 1000000);
  for (int i = 0; i < n; ++i) {
    data.push_back(dist(rng));
  }

  auto shell_sort = [](std::vector<int> &a) {
    size_t N = a.size();
    size_t gap = 1;
    while (gap < N / 3) {
      gap = gap * 3 + 1;
    }
    while (gap > 0) {
      for (size_t i = gap; i < N; ++i) {
        int tmp = a[i];
        size_t j = i;
        while (j >= gap && a[j - gap] > tmp) {
          a[j] = a[j - gap];
          j -= gap;
        }
        a[j] = tmp;
      }
      gap = (gap - 1) / 3;
    }
  };

  shell_sort(data);
  std::vector<int> left, right, merged;
  size_t mid = data.size() / 2;
  left.assign(data.begin(), data.begin() + mid);
  right.assign(data.begin() + mid, data.end());
  merged.resize(left.size() + right.size());
  std::merge(left.begin(), left.end(), right.begin(), right.end(), merged.begin());
  for (int phase = 0; phase < 2; ++phase) {
    size_t start = static_cast<size_t>(phase);
    for (size_t i = start; i + 1 < merged.size(); i += 2) {
      if (merged[i] > merged[i + 1]) {
        std::swap(merged[i], merged[i + 1]);
      }
    }
  }
  shell_sort(merged);
  long long checksum = 0;
  for (int v : merged) {
    checksum += v;
  }
  return checksum & 0x7FFFFFFF;
}

class YurkinGShellBetcherFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {}

  bool CheckTestOutputData(OutType &output_data) final {
    const int in = input_data_;
    long long expected = ComputeExpectedChecksumSeq(in);
    return static_cast<long long>(output_data) == expected;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(YurkinGShellBetcherFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<YurkinGShellBetcherMPI, InType>(kTestParam, PPC_SETTINGS_yurkin_g_shellbetcher),
    ppc::util::AddFuncTask<YurkinGShellBetcherSEQ, InType>(kTestParam, PPC_SETTINGS_yurkin_g_shellbetcher));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = YurkinGShellBetcherFuncTests::PrintFuncTestName<YurkinGShellBetcherFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, YurkinGShellBetcherFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace  yurkin_g_shellbetcher
