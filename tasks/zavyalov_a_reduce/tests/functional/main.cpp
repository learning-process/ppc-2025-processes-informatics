#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>
#include <mpi.h>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zavyalov_a_reduce/common/include/common.hpp"
#include "zavyalov_a_reduce/mpi/include/ops_mpi.hpp"
#include "zavyalov_a_reduce/seq/include/ops_seq.hpp"

namespace zavyalov_a_reduce {

class ZavyalovAReduceFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string mpi_typename, mpi_opname;
    if (std::get<0>(test_param) == MPI_SUM) { // TODO дописать другие операции в if
      mpi_opname = "sum";
    }
    else {
      throw "unsupported operation";
    }

    if (std::get<1>(test_param) == MPI_INT) { // TODO дописать другие типы в if
      mpi_typename = "int";
    }
    else {
      throw "unsupported datatype";
    }

    return mpi_opname + "_" + mpi_typename + "_" + std::to_string(std::get<2>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    unsigned int elems_count = std::get<2>(params);
    void* inp_data = nullptr;
    if (std::get<1>(params) == MPI_INT) {
      inp_data = new int[elems_count];
    }
    
    for (unsigned int i = 0; i < elems_count; i++) {
      ((int*)(inp_data))[i] = static_cast<int>(i) + 3U;
    }

    input_data_ = std::make_tuple(std::get<0>(params), std::get<1>(params), elems_count, inp_data);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    size_t vec_size = std::get<2>(params);
    void* res = new int[vec_size]; // TODO тут не всегда int[], в общем случае T[]
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    for (size_t i = 0; i < vec_size; i++) {
      ((int*)(res))[i] = ((int*)(std::get<3>(input_data_)))[i] * world_size; // TODO это только для MPI_SUM, для других тоже надо ифнуть
    }
    bool ok = true;
    for (size_t i = 0; i < vec_size; i++) {
      if (((int*)(res))[i] != ((int*)(output_data))[i]) {
        ok = false;
        break;
      }
    }
    
    return ok;
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

const std::array<TestType, 1> kTestParam = {std::make_tuple(MPI_SUM, MPI_INT, 5U)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ZavyalovAReduceMPI, InType>(kTestParam, PPC_SETTINGS_zavyalov_a_reduce),
    ppc::util::AddFuncTask<ZavyalovAReduceSEQ, InType>(kTestParam, PPC_SETTINGS_zavyalov_a_reduce));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ZavyalovAReduceFuncTests::PrintFuncTestName<ZavyalovAReduceFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, ZavyalovAReduceFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zavyalov_a_reduce
