#include <gtest/gtest.h>
#include <stb/stb_image.h>

// #include <algorithm>
#include <array>
#include <cstddef>
// #include <cstdint>
// #include <numeric>
// #include <stdexcept>
#include <string>
#include <tuple>
// #include <utility>
// #include <vector>

#include "konstantinov_s_elem_vec_sign_change_count/common/include/common.hpp"
#include "konstantinov_s_elem_vec_sign_change_count/mpi/include/ops_mpi.hpp"
#include "konstantinov_s_elem_vec_sign_change_count/seq/include/ops_seq.hpp"
#include "konstantinov_s_elem_vec_sign_change_count/tests/testgen.h"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace konstantinov_s_elem_vec_sign_change_count {

class KonstantinovSElemVecSignChangeTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    InType vec(params);
    vec.resize(params);
    std::array<EType, 15> arr = {1, -1, 23, -11, -12, -167, 13, 42, -12, 2, -43, 33, 44, -7, 1};
    const int arrsz = 15;
    const int chngcnt = 10;

    result_right_ = GenerateTestData(arr.data(), arrsz, chngcnt, vec);
    input_data_ = vec;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // std::cout<<"CHECK: ";
    // for(int i=0;i<input_data_.size();i++)
    //   std::cout<<input_data_[i]<<" ";
    // std::cout<<"CHECK DATA: "<<output_data<<" ?= "<<result_right<<"\n";
    return (output_data == result_right_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType result_right_{};
};

namespace {

TEST_P(KonstantinovSElemVecSignChangeTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {1, 7, 15, 30, 33};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KonstantinovSElemVecSignChangeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_konstantinov_s_elem_vec_sign_change_count),
                                           ppc::util::AddFuncTask<KonstantinovSElemVecSignChangeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_konstantinov_s_elem_vec_sign_change_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KonstantinovSElemVecSignChangeTests::PrintFuncTestName<KonstantinovSElemVecSignChangeTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KonstantinovSElemVecSignChangeTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace konstantinov_s_elem_vec_sign_change_count
