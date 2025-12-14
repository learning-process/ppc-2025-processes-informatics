#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <string>
#include <tuple>
#include <vector>

#include "romanov_a_scatter/common/include/common.hpp"
#include "romanov_a_scatter/mpi/include/ops_mpi.hpp"
#include "romanov_a_scatter/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace romanov_a_scatter {

class RomanovAScatterFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    auto [sendbuf, sendcount, root] = test_param;

    std::string test_param_str;
    test_param_str += "_sendbuf";
    ;
    for (const auto &val : sendbuf) {
      test_param_str += "_" + std::to_string(val);
    }
    test_param_str += "_sendcount" + std::to_string(sendcount);
    test_param_str += "_root" + std::to_string(root);
    return test_param_str;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int root = std::get<2>(params);
    int sendcount = std::get<1>(params);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_processes = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    std::vector<int> sendbuf;
    if (rank == root) {
      sendbuf = std::get<0>(params);
      sendbuf.resize(num_processes * sendcount);
    }

    input_data_ = std::make_tuple(sendbuf, sendcount, root);

    std::vector<int> recvbuf(sendcount);
    MPI_Scatter(sendbuf.data(), sendcount, MPI_INT, recvbuf.data(), sendcount, MPI_INT, root, MPI_COMM_WORLD);

    expected_ = recvbuf;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

namespace {

TEST_P(RomanovAScatterFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {
    // Фреймворк не позволяет установить root != 0 так, чтобы
    // гарантировать существование необходимого числа процессов
    std::make_tuple(std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 0),
    std::make_tuple(std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 4, 0),
    std::make_tuple(std::vector<int>{0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 11}, 8, 0),
    std::make_tuple(std::vector<int>{12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}, 16, 0),
    std::make_tuple(std::vector<int>{1, 3, 2, 4}, 64, 0)};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<RomanovAScatterMPI, InType>(kTestParam, PPC_SETTINGS_romanov_a_scatter),
                   ppc::util::AddFuncTask<RomanovAScatterSEQ, InType>(kTestParam, PPC_SETTINGS_romanov_a_scatter));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = RomanovAScatterFuncTests::PrintFuncTestName<RomanovAScatterFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, RomanovAScatterFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace romanov_a_scatter
