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

#include "telnov_transfer_one_all/common/include/common.hpp"
#include "telnov_transfer_one_all/mpi/include/ops_mpi.hpp"
#include "telnov_transfer_one_all/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace telnov_transfer_one_all {

using TestType = std::tuple<int, std::string>;
using InTypeInt = std::vector<int>;
using InTypeFloat = std::vector<float>;
using InTypeDouble = std::vector<double>;
using OutTypeInt = InTypeInt;
using OutTypeFloat = InTypeFloat;
using OutTypeDouble = InTypeDouble;

class TelnovTransferOneAllFuncTests : public ppc::util::BaseRunFuncTests<InTypeInt, OutTypeInt, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const int array_size = 10;
    input_data_.resize(array_size);

    for (int i = 0; i < array_size; ++i) {
      input_data_[i] = std::rand() % 100;
    }

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size_);
    root_ = std::rand() % comm_size_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::equal(input_data_.begin(), input_data_.end(), output_data.begin());
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(TelnovTransferOneAllFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TelnovTransferOneAllMPI, InType>(kTestParam, PPC_SETTINGS_telnov_transfer_one_all),
    ppc::util::AddFuncTask<TelnovTransferOneAllSEQ, InType>(kTestParam, PPC_SETTINGS_telnov_transfer_one_all));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TelnovTransferOneAllFuncTests::PrintFuncTestName<TelnovTransferOneAllFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, TelnovTransferOneAllFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace telnov_transfer_one_all
