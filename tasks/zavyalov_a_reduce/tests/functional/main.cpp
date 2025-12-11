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
   ~ZavyalovAReduceFuncTests() override { // для освобождения выделенной памяти
    CleanupInputData();
  }
  static std::string PrintTestParam(const TestType &test_param) {
    std::string mpi_typename, mpi_opname;
    if (std::get<0>(test_param) == MPI_SUM) { // TODO дописать другие операции в if
      mpi_opname = "sum";
    } else if (std::get<0>(test_param) == MPI_MIN){
      mpi_opname = "min";
    } else {
      std::cout << "unsupported operation\n";
      throw "unsupported operation";
    }

    if (std::get<1>(test_param) == MPI_INT) { // TODO дописать другие типы в if
      mpi_typename = "int";
    } else if (std::get<1>(test_param) == MPI_FLOAT ) {
      mpi_typename = "float";
    } else if (std::get<1>(test_param) == MPI_DOUBLE ) {
      mpi_typename = "double";
    } else {
      std::cout << "unsupported datatype\n"; 
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

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    void* inp_data = nullptr;

    if (cur_type == MPI_INT) {
      inp_data = new int[vec_size];
    } else if (cur_type == MPI_FLOAT) {
      inp_data = new float[vec_size];
    } else if (cur_type == MPI_DOUBLE) {
      inp_data = new double[vec_size];
    }

    if (operation == MPI_SUM) {
      for (unsigned int i = 0; i < vec_size; i++) {
        if (cur_type == MPI_INT) {
          ((int*)(inp_data))[i] = static_cast<int>(i) + 3U;
        } else if (cur_type == MPI_FLOAT) {
          ((float*)(inp_data))[i] = static_cast<float>(i) + 3.0;
        } else if (cur_type == MPI_DOUBLE) {
          ((double*)(inp_data))[i] = static_cast<double>(i) + 3.0;
        }
      }
    } else if (operation == MPI_MIN) {
      for (unsigned int i = 0; i < vec_size; i++) {
        if (cur_type == MPI_INT) {
          ((int*)(inp_data))[i] = 10000u - 3u * rank;
        } else if (cur_type == MPI_FLOAT) {
          ((float*)(inp_data))[i] = 10000.0 - 3.0 * static_cast<float>(rank);
        } else if (cur_type == MPI_DOUBLE) {
          ((float*)(inp_data))[i] = 10000.0 - 3.0 * static_cast<double>(rank);
        }
      }
    }

    input_data_ = std::make_tuple(operation, cur_type, vec_size, inp_data, std::get<3>(params));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data == nullptr) {
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
    std::cout << "### world_size in CheckTestOutputData: " << world_size << " ###" << std::endl;

    if (operation == MPI_SUM) {
      if (cur_type == MPI_INT) {
        std::vector<int> res(vec_size);
        for (size_t i = 0; i < vec_size; i++) {
          res[i] = ((int*)(std::get<3>(input_data_)))[i] * world_size; // TODO это только для MPI_SUM, для других тоже надо ифнуть
          std::cout << res[i] << ' ';
        }
        std::cout << "- actual res \n";

        for (size_t i = 0; i < vec_size; i++) {
          if (res[i] != ((int*)(output_data))[i]) {
            ok = false;
            std::cout << "i = " << i <<  ", отличие тут. res[i]= " << res[i] << ", output_data[i]= " << ((int*)(output_data))[i] << std::endl;
            break;
          }
        }
      } else if (cur_type == MPI_FLOAT) {
        std::vector<float> res(vec_size);
        for (size_t i = 0; i < vec_size; i++) {
          res[i] = ((float*)(std::get<3>(input_data_)))[i] * world_size; // TODO это только для MPI_SUM, для других тоже надо ифнуть
          std::cout << res[i] << ' ';
        }

        for (size_t i = 0; i < vec_size; i++) {
          if (res[i] != ((float*)(output_data))[i]) {
            ok = false;
            std::cout << "i = " << i <<  ", отличие тут. res[i]= " << res[i] << ", output_data[i]= " << ((float*)(output_data))[i] << std::endl;
            break;
          }
        }
      } else if (cur_type == MPI_DOUBLE) {
        std::vector<double> res(vec_size);
        for (size_t i = 0; i < vec_size; i++) {
          res[i] = ((double*)(std::get<3>(input_data_)))[i] * world_size; // TODO это только для MPI_SUM, для других тоже надо ифнуть
          std::cout << res[i] << ' ';
        }

        for (size_t i = 0; i < vec_size; i++) {
          if (res[i] != ((double*)(output_data))[i]) {
            ok = false;
            std::cout << "i = " << i <<  ", отличие тут. res[i]= " << res[i] << ", output_data[i]= " << ((double*)(output_data))[i] << std::endl;
            break;
          }
        }
      }
    } else if (operation == MPI_MIN) {
      if (cur_type == MPI_INT) {
        std::vector<int> res(vec_size);
        for (size_t i = 0; i < vec_size; i++) {
          res[i] = ((int*)(std::get<3>(input_data_)))[i] * world_size; // TODO это только для MPI_SUM, для других тоже надо ифнуть
          std::cout << res[i] << ' ';
        }
        std::cout << "- actual res \n";

        for (size_t i = 0; i < vec_size; i++) {
          if (res[i] != ((int*)(output_data))[i]) {
            ok = false;
            std::cout << "i = " << i <<  ", отличие тут. res[i]= " << res[i] << ", output_data[i]= " << ((int*)(output_data))[i] << std::endl;
            break;
          }
        }
      } else if (cur_type == MPI_FLOAT) {

      } else if (cur_type == MPI_DOUBLE) {

      }
    }

    

    return ok;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  void CleanupInputData() {
    void* data = std::get<3>(input_data_);
    if (data != nullptr) {
      // Нужно знать тип данных для правильного удаления
      TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
      if (std::get<1>(params) == MPI_INT) {
        delete[] static_cast<int*>(data);
      } else if (std::get<1>(params) == MPI_FLOAT) {
        delete[] static_cast<float*>(data);
      } else if (std::get<1>(params) == MPI_DOUBLE) {
        delete[] static_cast<double*>(data);
      }

      // TODO: обработка других типов данных
      std::get<3>(input_data_) = nullptr;
    }
  }
};

namespace {

TEST_P(ZavyalovAReduceFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
  std::make_tuple(MPI_SUM, MPI_INT, 5U, 0),
  std::make_tuple(MPI_SUM, MPI_INT, 9U, 1),
  std::make_tuple(MPI_SUM, MPI_FLOAT, 11U, 0)
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ZavyalovAReduceMPI, InType>(kTestParam, PPC_SETTINGS_zavyalov_a_reduce),
    ppc::util::AddFuncTask<ZavyalovAReduceSEQ, InType>(kTestParam, PPC_SETTINGS_zavyalov_a_reduce));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ZavyalovAReduceFuncTests::PrintFuncTestName<ZavyalovAReduceFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, ZavyalovAReduceFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zavyalov_a_reduce
