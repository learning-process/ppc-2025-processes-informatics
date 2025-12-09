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

#include "baldin_a_my_scatter/common/include/common.hpp"
#include "baldin_a_my_scatter/mpi/include/ops_mpi.hpp"
#include "baldin_a_my_scatter/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace baldin_a_my_scatter {

class BaldinAMyScatterFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    auto [count, root, type] = test_param;
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string type_str = (type == MPI_INT ? "INT" : (type == MPI_FLOAT ? "FLOAT" : "DOUBLE"));
    return type_str + "_Count" + std::to_string(count) + "_Root" + std::to_string(root) + "_RRoot" + std::to_string(root % size);
  }

 protected:
  
  std::vector<int> send_vec_int_, recv_vec_int_;
  std::vector<float> send_vec_float_, recv_vec_float_;
  std::vector<double> send_vec_double_, recv_vec_double_;

  InType input_data_;
  
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int count = std::get<0>(params);
    int root = std::get<1>(params);
    MPI_Datatype type = std::get<2>(params);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    root = root % size;

    const void* sendbuf_ptr = nullptr;
    void* recvbuf_ptr = nullptr;

    if (type == MPI_INT) {
        recv_vec_int_.resize(count);
        recvbuf_ptr = recv_vec_int_.data();

        if (rank == root) {
            send_vec_int_.resize(count * size);
            std::iota(send_vec_int_.begin(), send_vec_int_.end(), 0);
            sendbuf_ptr = send_vec_int_.data();
        }
    } 
    else if (type == MPI_FLOAT) {
        recv_vec_float_.resize(count);
        recvbuf_ptr = recv_vec_float_.data();

        if (rank == root) {
            send_vec_float_.resize(count * size);
            for(int i=0; i < count*size; ++i) send_vec_float_[i] = static_cast<float>(i);
            sendbuf_ptr = send_vec_float_.data();
        }
    }
    else if (type == MPI_DOUBLE) {
        recv_vec_double_.resize(count);
        recvbuf_ptr = recv_vec_double_.data();

        if (rank == root) {
            send_vec_double_.resize(count * size);
            for(int i=0; i < count*size; ++i) send_vec_double_[i] = static_cast<double>(i);
            sendbuf_ptr = send_vec_double_.data();
        }
    }

    input_data_ = std::make_tuple(sendbuf_ptr, count, type, recvbuf_ptr, count, type, root, MPI_COMM_WORLD);
  }

  bool CheckTestOutputData(OutType &output_data) final {

    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int count = std::get<0>(params);
    MPI_Datatype type = std::get<2>(params);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (output_data == nullptr) return false;

    int start_value = rank * count;
    if (type == MPI_INT) {
        const int* actual_data = reinterpret_cast<const int*>(output_data);
        for (int i = 0; i < count; i++) {
            if (actual_data[i] != start_value + i) {
                return false;
            }
            //std::cout << rank << "i:" << actual_data[i] << ' ';
        }
    } else if (type == MPI_FLOAT) {
        const float* actual_data = reinterpret_cast<const float*>(output_data);
        for (int i = 0; i < count; i++) {
            if (std::abs(actual_data[i] - (float)(start_value + i)) >= 1e-6) {
                return false;
            }
            //std::cout << rank << "f:" << actual_data[i] << ' ';
        }
    } else if (type == MPI_DOUBLE) {
        const double* actual_data = reinterpret_cast<const double*>(output_data);
        for (int i = 0; i < count; i++) {
            if (std::abs(actual_data[i] - (double)(start_value + i)) >= 1e-10) {
                return false;
            }
            //std::cout << rank << "d:" << actual_data[i] << ' ';
        }
    } else {
        return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
  
};

namespace {

TEST_P(BaldinAMyScatterFuncTests, MyScatterTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 19> kTestParam = { 
    std::make_tuple(1, 0, MPI_INT),

    std::make_tuple(10, 0, MPI_INT),
    std::make_tuple(10, 0, MPI_FLOAT),
    std::make_tuple(10, 0, MPI_DOUBLE),

    std::make_tuple(10, 1, MPI_INT),
    std::make_tuple(10, 1, MPI_FLOAT),
    std::make_tuple(10, 1, MPI_DOUBLE),

    std::make_tuple(10, 2, MPI_INT),
    std::make_tuple(10, 2, MPI_FLOAT),
    std::make_tuple(10, 2, MPI_DOUBLE),

    std::make_tuple(10, 3, MPI_INT),
    std::make_tuple(10, 3, MPI_FLOAT),
    std::make_tuple(10, 3, MPI_DOUBLE),

    std::make_tuple(17, 0, MPI_INT),
    std::make_tuple(123, 0, MPI_INT),
    std::make_tuple(7, 1, MPI_DOUBLE),
    
    std::make_tuple(1000, 0, MPI_INT),
    std::make_tuple(500, 1, MPI_DOUBLE),
    std::make_tuple(1500, 2, MPI_FLOAT)
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<BaldinAMyScatterMPI, InType>(kTestParam, PPC_SETTINGS_baldin_a_my_scatter),
                   ppc::util::AddFuncTask<BaldinAMyScatterSEQ, InType>(kTestParam, PPC_SETTINGS_baldin_a_my_scatter));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BaldinAMyScatterFuncTests::PrintFuncTestName<BaldinAMyScatterFuncTests>;

INSTANTIATE_TEST_SUITE_P(MyScatterTests, BaldinAMyScatterFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace baldin_a_my_scatter