#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <random>
#include <ranges>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yurkin_g_shellbetcher/common/include/common.hpp"
#include "yurkin_g_shellbetcher/mpi/include/ops_mpi.hpp"
#include "yurkin_g_shellbetcher/seq/include/ops_seq.hpp"

// Forward declaration to make BatcherMerge visible to tests without changing headers
namespace yurkin_g_shellbetcher {
void BatcherMerge(const std::vector<int> &left, const std::vector<int> &right, std::vector<int> &out);
}

namespace yurkin_g_shellbetcher {

static std::int64_t ComputeExpectedChecksumSeq(int n) {
  std::vector<int> data;
  data.reserve(static_cast<std::size_t>(n));
  std::mt19937 rng(static_cast<unsigned int>(n));
  std::uniform_int_distribution<int> dist(0, 1000000);
  for (int i = 0; i < n; ++i) {
    data.push_back(dist(rng));
  }
  std::ranges::sort(data);
  std::int64_t checksum = 0;
  for (int v : data) {
    checksum += static_cast<std::int64_t>(v);
  }
  return checksum & 0x7FFFFFFF;
}

TEST(BatcherMergeUnitTests, MergesTwoSortedVectorsCorrectly) {
  std::vector<int> a = {1, 3, 5, 7};
  std::vector<int> b = {2, 4, 6, 8, 9};
  std::vector<int> out;
  yurkin_g_shellbetcher::BatcherMerge(a, b, out);
  ASSERT_EQ(out.size(), a.size() + b.size());
  ASSERT_TRUE(std::ranges::is_sorted(out));
  std::vector<int> expected;
  expected.reserve(out.size());
  std::ranges::merge(a, b, std::back_inserter(expected));
  ASSERT_EQ(out, expected);
}

TEST(BatcherMergeUnitTests, HandlesEmptyInputs) {
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> out;
  yurkin_g_shellbetcher::BatcherMerge(a, b, out);
  ASSERT_TRUE(out.empty());
}

TEST(BatcherMergeUnitTests, HandlesDuplicatesAndOddSizes) {
  std::vector<int> a = {1, 1, 3, 5};
  std::vector<int> b = {1, 2, 2};
  std::vector<int> out;
  yurkin_g_shellbetcher::BatcherMerge(a, b, out);
  ASSERT_EQ(out.size(), a.size() + b.size());
  ASSERT_TRUE(std::ranges::is_sorted(out));
  std::vector<int> expected;
  std::ranges::merge(a, b, std::back_inserter(expected));
  ASSERT_EQ(out, expected);
}

TEST(BatcherMergeUnitTests, LargeRandomVectorsMatchStdMerge) {
  std::mt19937 rng(std::random_device{}());  // NOLINT(cert-msc51-cpp)
  std::uniform_int_distribution<int> dist(0, 1000);
  std::vector<int> a(1000);
  std::vector<int> b(1500);
  for (auto &v : a) {
    v = dist(rng);
  }
  for (auto &v : b) {
    v = dist(rng);
  }
  std::ranges::sort(a);
  std::ranges::sort(b);
  std::vector<int> out;
  yurkin_g_shellbetcher::BatcherMerge(a, b, out);
  ASSERT_EQ(out.size(), a.size() + b.size());
  ASSERT_TRUE(std::ranges::is_sorted(out));
  std::vector<int> expected;
  expected.reserve(out.size());
  std::ranges::merge(a, b, std::back_inserter(expected));
  ASSERT_EQ(out, expected);
}

class YurkinGShellBetcherFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = static_cast<InType>(std::get<0>(params));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const int in = input_data_;
    std::int64_t expected = ComputeExpectedChecksumSeq(in);
    return static_cast<std::int64_t>(output_data) == expected;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

TEST_P(YurkinGShellBetcherFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<YurkinGShellBetcherMPI, InType>(kTestParam, PPC_SETTINGS_yurkin_g_shellbetcher),
    ppc::util::AddFuncTask<YurkinGShellBetcherSEQ, InType>(kTestParam, PPC_SETTINGS_yurkin_g_shellbetcher));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, YurkinGShellBetcherFuncTests, kGtestValues,
                         YurkinGShellBetcherFuncTests::PrintFuncTestName<YurkinGShellBetcherFuncTests>);

}  // namespace yurkin_g_shellbetcher
