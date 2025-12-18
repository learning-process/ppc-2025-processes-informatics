#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zavyalov_a_qsort_simple_merge/common/include/common.hpp"
#include "zavyalov_a_qsort_simple_merge/mpi/include/ops_mpi.hpp"
#include "zavyalov_a_qsort_simple_merge/seq/include/ops_seq.hpp"

namespace zavyalov_a_qsort_simple_merge {

class ZavyalovAReduceFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    size_t vec_size = params;

    std::srand(std::time({}));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 99);

    std::vector<double> vec(vec_size);
    double mult = 1;
    for (size_t i = 0U; i < vec_size; i++) {
      vec[i] = distrib(gen) * mult;
      mult *= -1.0;
    }

    input_data_ = vec;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto vec = input_data_;
    std::ranges::sort(vec);
    for (size_t i = 0; i < vec.size(); i++) {
      if (vec[i] != output_data[i]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(ZavyalovAReduceFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 12> kTestParam = {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 100U, 1000U};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ZavyalovAQsortMPI, InType>(kTestParam, PPC_SETTINGS_zavyalov_a_qsort_simple_merge),
    ppc::util::AddFuncTask<ZavyalovAQsortSEQ, InType>(kTestParam, PPC_SETTINGS_zavyalov_a_qsort_simple_merge));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ZavyalovAReduceFuncTests::PrintFuncTestName<ZavyalovAReduceFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, ZavyalovAReduceFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zavyalov_a_qsort_simple_merge
