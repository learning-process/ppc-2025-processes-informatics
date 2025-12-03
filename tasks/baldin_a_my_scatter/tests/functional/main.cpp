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
    const auto& [sendbuf, sendcount, sendtype, recvbuf_dummy, recvcount, recvtype, root, comm] = test_param;

    int world_size = 0;
    MPI_Comm_size(comm, &world_size);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::string type = (sendtype == MPI_INT ? "int_" : (sendtype == MPI_FLOAT ? "float_" : "double_"));
    return std::to_string(world_size) + "proc_" + std::to_string(world_size * sendcount) + type + std::to_string(root % world_size) + "root_" + std::to_string(root) + "rroot";
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = params;
    expected_output_ = nullptr;

  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto& input = GetTestInputData();
    const auto& [sendbuf, sendcount, sendtype, recvbuf_dummy, recvcount, recvtype, root, comm] = input;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    
    if (recvcount > 0 && output_data == nullptr) {
        return false;
    }

    int start_value = rank * recvcount + 1;
    if (recvtype == MPI_INT) {
        const int* actual_data = reinterpret_cast<const int*>(output_data);
        for (int i = 0; i < recvcount; i++) {
            if (actual_data[i] != start_value + i) {
                return false;
            }
            //std::cout << rank << "i:" << actual_data[i] << ' ';
        }
    } else if (recvtype == MPI_FLOAT) {
        const float* actual_data = reinterpret_cast<const float*>(output_data);
        for (int i = 0; i < recvcount; i++) {
            if (std::abs(actual_data[i] - (float)(start_value + i)) >= 1e-6) {
                return false;
            }
            //std::cout << rank << "f:" << actual_data[i] << ' ';
        }
    } else if (recvtype == MPI_DOUBLE) {
        const double* actual_data = reinterpret_cast<const double*>(output_data);
        for (int i = 0; i < recvcount; i++) {
            if (std::abs(actual_data[i] - (double)(start_value + i)) >= 1e-6) {
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

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(BaldinAMyScatterFuncTests, MyScatterTests) {
  ExecuteTest(GetParam());
}

int send_data1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
int recv_data1[3] = {};

InType test1 = std::make_tuple(
    static_cast<const void*>(send_data1), // 1. sendbuf (const void*)
    3,                                    // 2. sendcount (int)
    MPI_INT,                              // 3. sendtype (MPI_Datatype)
    static_cast<void*>(recv_data1),       // 4. recvbuf (void *)
    3,                                    // 5. recvcount (int)
    MPI_INT,                              // 6. recvtype (MPI_Datatype)
    0,                                    // 7. root (int)
    MPI_COMM_WORLD                        // 8. comm (MPI_Comm)
);

float send_data2[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
float recv_data2[3] = {};

InType test2 = std::make_tuple(
    static_cast<const void*>(send_data2),
    3,
    MPI_FLOAT,
    static_cast<void*>(recv_data2),
    3,
    MPI_FLOAT,
    0,
    MPI_COMM_WORLD
);

double send_data3[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
double recv_data3[3] = {};

InType test3 = std::make_tuple(
    static_cast<const void*>(send_data3),
    3,
    MPI_DOUBLE,
    static_cast<void*>(recv_data3),
    3,
    MPI_DOUBLE,
    0,
    MPI_COMM_WORLD
);

int send_data4[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
int recv_data4[3] = {};

InType test4 = std::make_tuple(
    static_cast<const void*>(send_data4), // 1. sendbuf (const void*)
    3,                                    // 2. sendcount (int)
    MPI_INT,                              // 3. sendtype (MPI_Datatype)
    static_cast<void*>(recv_data4),       // 4. recvbuf (void *)
    3,                                    // 5. recvcount (int)
    MPI_INT,                              // 6. recvtype (MPI_Datatype)
    1,                                    // 7. root (int)
    MPI_COMM_WORLD                        // 8. comm (MPI_Comm)
);

float send_data5[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
float recv_data5[3] = {};

InType test5 = std::make_tuple(
    static_cast<const void*>(send_data5),
    3,
    MPI_FLOAT,
    static_cast<void*>(recv_data5),
    3,
    MPI_FLOAT,
    1,
    MPI_COMM_WORLD
);

double send_data6[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
double recv_data6[3] = {};

InType test6 = std::make_tuple(
    static_cast<const void*>(send_data6),
    3,
    MPI_DOUBLE,
    static_cast<void*>(recv_data6),
    3,
    MPI_DOUBLE,
    1,
    MPI_COMM_WORLD
);

int send_data7[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
int recv_data7[3] = {};

InType test7 = std::make_tuple(
    static_cast<const void*>(send_data7), // 1. sendbuf (const void*)
    3,                                    // 2. sendcount (int)
    MPI_INT,                              // 3. sendtype (MPI_Datatype)
    static_cast<void*>(recv_data7),       // 4. recvbuf (void *)
    3,                                    // 5. recvcount (int)
    MPI_INT,                              // 6. recvtype (MPI_Datatype)
    2,                                    // 7. root (int)
    MPI_COMM_WORLD                        // 8. comm (MPI_Comm)
);

float send_data8[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
float recv_data8[3] = {};

InType test8 = std::make_tuple(
    static_cast<const void*>(send_data8),
    3,
    MPI_FLOAT,
    static_cast<void*>(recv_data8),
    3,
    MPI_FLOAT,
    2,
    MPI_COMM_WORLD
);

double send_data9[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
double recv_data9[3] = {};

InType test9 = std::make_tuple(
    static_cast<const void*>(send_data9),
    3,
    MPI_DOUBLE,
    static_cast<void*>(recv_data9),
    3,
    MPI_DOUBLE,
    2,
    MPI_COMM_WORLD
);

int send_data10[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
int recv_data10[3] = {};

InType test10 = std::make_tuple(
    static_cast<const void*>(send_data10), // 1. sendbuf (const void*)
    3,                                     // 2. sendcount (int)
    MPI_INT,                               // 3. sendtype (MPI_Datatype)
    static_cast<void*>(recv_data10),       // 4. recvbuf (void *)
    3,                                     // 5. recvcount (int)
    MPI_INT,                               // 6. recvtype (MPI_Datatype)
    3,                                     // 7. root (int)
    MPI_COMM_WORLD                         // 8. comm (MPI_Comm)
);

float send_data11[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
float recv_data11[3] = {};

InType test11 = std::make_tuple(
    static_cast<const void*>(send_data11),
    3,
    MPI_FLOAT,
    static_cast<void*>(recv_data11),
    3,
    MPI_FLOAT,
    3,
    MPI_COMM_WORLD
);

double send_data12[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
double recv_data12[3] = {};

InType test12 = std::make_tuple(
    static_cast<const void*>(send_data12),
    3,
    MPI_DOUBLE,
    static_cast<void*>(recv_data12),
    3,
    MPI_DOUBLE,
    3,
    MPI_COMM_WORLD
);

const std::array<TestType, 12> kTestParam = { 
    // отправка с rank = 0 (тип int, float, double)
    test1, test2, test3,

    // отправка с rank = 1
    test4, test5, test6,

    // отправка с rank = 2
    test7, test8, test9,

    // отправка с rank = 3
    test10, test11, test12
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<BaldinAMyScatterMPI, InType>(kTestParam, PPC_SETTINGS_baldin_a_my_scatter),
                   ppc::util::AddFuncTask<BaldinAMyScatterSEQ, InType>(kTestParam, PPC_SETTINGS_baldin_a_my_scatter));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BaldinAMyScatterFuncTests::PrintFuncTestName<BaldinAMyScatterFuncTests>;

INSTANTIATE_TEST_SUITE_P(MyScatterTests, BaldinAMyScatterFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace baldin_a_my_scatter