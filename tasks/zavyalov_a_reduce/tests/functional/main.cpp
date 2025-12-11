#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zavyalov_a_reduce/common/include/common.hpp"
#include "zavyalov_a_reduce/mpi/include/ops_mpi.hpp"
#include "zavyalov_a_reduce/seq/include/ops_seq.hpp"

namespace zavyalov_a_reduce {

class ZavyalovAReduceFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  ~ZavyalovAReduceFuncTests() override = default;  // shared_ptr автоматически управляет памятью

  static std::string PrintTestParam(const TestType &test_param) {
    std::string mpi_opname;
    std::string mpi_typename;
    if (std::get<0>(test_param) == MPI_SUM) {
      mpi_opname = "sum";
    } else if (std::get<0>(test_param) == MPI_MIN) {
      mpi_opname = "min";
    } else {
      throw "unsupported operation";
    }

    if (std::get<1>(test_param) == MPI_INT) {
      mpi_typename = "int";
    } else if (std::get<1>(test_param) == MPI_FLOAT) {
      mpi_typename = "float";
    } else if (std::get<1>(test_param) == MPI_DOUBLE) {
      mpi_typename = "double";
    } else {
      throw "unsupported datatype";
    }

    return mpi_opname + "_" + mpi_typename + "_" + std::to_string(std::get<2>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    MPI_Op operation = std::get<0>(params);
    MPI_Datatype cur_type = std::get<1>(params);
    size_t vec_size = std::get<2>(params);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::shared_ptr<void> inp_data;

    if (cur_type == MPI_INT) {
      int *raw_data = new int[vec_size];
      inp_data = std::shared_ptr<void>(raw_data, [](void *p) { delete[] static_cast<int *>(p); });
    } else if (cur_type == MPI_FLOAT) {
      float *raw_data = new float[vec_size];
      inp_data = std::shared_ptr<void>(raw_data, [](void *p) { delete[] static_cast<float *>(p); });
    } else if (cur_type == MPI_DOUBLE) {
      double *raw_data = new double[vec_size];
      inp_data = std::shared_ptr<void>(raw_data, [](void *p) { delete[] static_cast<double *>(p); });
    } else {
      throw "unsupported datatype";
    }

    if (operation == MPI_SUM) {
      for (unsigned int i = 0; i < vec_size; i++) {
        if (cur_type == MPI_INT) {
          (static_cast<int *>(inp_data.get()))[i] = static_cast<int>(i) + 3;
        } else if (cur_type == MPI_FLOAT) {
          (static_cast<float *>(inp_data.get()))[i] = (static_cast<float>(i) * 1.1f) + 3.0F;
        } else if (cur_type == MPI_DOUBLE) {
          (static_cast<double *>(inp_data.get()))[i] = (static_cast<double>(i) * 1.1) + 3.0;
        }
      }
    } else if (operation == MPI_MIN) {
      for (unsigned int i = 0; i < vec_size; i++) {
        if (cur_type == MPI_INT) {
          (static_cast<int *>(inp_data.get()))[i] = 10000U - (3U * rank);
        } else if (cur_type == MPI_FLOAT) {
          (static_cast<float *>(inp_data.get()))[i] = 10000.0F - (3.0F * static_cast<float>(rank));
        } else if (cur_type == MPI_DOUBLE) {
          (static_cast<double *>(inp_data.get()))[i] = 10000.0 - (3.0 * static_cast<double>(rank));
        }
      }
    } else {
      throw "unsupported operation";
    }

    input_data_ = std::make_tuple(operation, cur_type, vec_size, inp_data, std::get<3>(params));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (std::get<1>(output_data)) {
      return true;
    }

    std::shared_ptr<void> result_ptr = std::get<0>(output_data);
    if (!result_ptr) {
      return false;
    }

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    MPI_Op operation = std::get<0>(params);
    MPI_Datatype cur_type = std::get<1>(params);
    size_t vec_size = std::get<2>(params);
    bool ok = true;

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Получаем входные данные из shared_ptr
    std::shared_ptr<void> input_ptr = std::get<3>(input_data_);

    if (operation == MPI_SUM) {
      if (cur_type == MPI_INT) {
        std::vector<int> res(vec_size);
        int *input_data = static_cast<int *>(input_ptr.get());
        for (size_t i = 0; i < vec_size; i++) {
          res[i] = input_data[i] * world_size;
        }

        int *result_data = static_cast<int *>(result_ptr.get());
        for (size_t i = 0; i < vec_size; i++) {
          if (res[i] != result_data[i]) {
            ok = false;
            break;
          }
        }
      } else if (cur_type == MPI_FLOAT) {
        std::vector<float> res(vec_size);
        float *input_data = static_cast<float *>(input_ptr.get());
        for (size_t i = 0; i < vec_size; i++) {
          res[i] = input_data[i] * static_cast<float>(world_size);
        }

        float *result_data = static_cast<float *>(result_ptr.get());
        for (size_t i = 0; i < vec_size; i++) {
          if (res[i] != result_data[i]) {
            ok = false;
            break;
          }
        }
      } else if (cur_type == MPI_DOUBLE) {
        std::vector<double> res(vec_size);
        double *input_data = static_cast<double *>(input_ptr.get());
        for (size_t i = 0; i < vec_size; i++) {
          res[i] = input_data[i] * static_cast<double>(world_size);
        }

        double *result_data = static_cast<double *>(result_ptr.get());
        for (size_t i = 0; i < vec_size; i++) {
          if (res[i] != result_data[i]) {
            ok = false;
            break;
          }
        }
      }
    } else if (operation == MPI_MIN) {
      if (cur_type == MPI_INT) {
        std::vector<int> res(vec_size);
        for (size_t i = 0; i < vec_size; i++) {
          res[i] = 10000 - (3 * (world_size - 1));
        }

        int *result_data = static_cast<int *>(result_ptr.get());
        for (size_t i = 0; i < vec_size; i++) {
          if (res[i] != result_data[i]) {
            ok = false;
            break;
          }
        }
      } else if (cur_type == MPI_FLOAT) {
        std::vector<float> res(vec_size);
        for (size_t i = 0; i < vec_size; i++) {
          res[i] = 10000.0F - (3.0F * static_cast<float>(world_size - 1));
        }

        float *result_data = static_cast<float *>(result_ptr.get());
        for (size_t i = 0; i < vec_size; i++) {
          if (res[i] != result_data[i]) {
            ok = false;
            break;
          }
        }
      } else if (cur_type == MPI_DOUBLE) {
        std::vector<double> res(vec_size);
        for (size_t i = 0; i < vec_size; i++) {
          res[i] = 10000.0 - (3.0 * static_cast<double>(world_size - 1));
        }

        double *result_data = static_cast<double *>(result_ptr.get());
        for (size_t i = 0; i < vec_size; i++) {
          if (res[i] != result_data[i]) {
            ok = false;
            break;
          }
        }
      }
    }

    // Не нужно удалять вручную - shared_ptr сам управляет памятью
    // Память автоматически освободится при выходе из scope

    return ok;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  // CleanupData больше не нужен - память управляется shared_ptr
};

namespace {

TEST_P(ZavyalovAReduceFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(MPI_SUM, MPI_INT, 5U, 0),    std::make_tuple(MPI_SUM, MPI_INT, 9U, 0),
    std::make_tuple(MPI_SUM, MPI_FLOAT, 11U, 0), std::make_tuple(MPI_SUM, MPI_DOUBLE, 10U, 0),
    std::make_tuple(MPI_MIN, MPI_FLOAT, 6U, 0),  std::make_tuple(MPI_MIN, MPI_FLOAT, 50U, 0),
    std::make_tuple(MPI_MIN, MPI_INT, 6U, 0),    std::make_tuple(MPI_MIN, MPI_DOUBLE, 6U, 0)};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<ZavyalovAReduceMPI, InType>(kTestParam, PPC_SETTINGS_zavyalov_a_reduce),
                   ppc::util::AddFuncTask<ZavyalovAReduceSEQ, InType>(kTestParam, PPC_SETTINGS_zavyalov_a_reduce));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ZavyalovAReduceFuncTests::PrintFuncTestName<ZavyalovAReduceFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, ZavyalovAReduceFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zavyalov_a_reduce
