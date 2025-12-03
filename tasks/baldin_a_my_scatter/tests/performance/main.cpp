// #include <gtest/gtest.h>

// #include "baldin_a_my_scatter/common/include/common.hpp"
// #include "baldin_a_my_scatter/mpi/include/ops_mpi.hpp"
// #include "baldin_a_my_scatter/seq/include/ops_seq.hpp"
// #include "util/include/perf_test_util.hpp"

// namespace baldin_a_my_scatter {

// class BaldinAMyScatterPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
//   InType input_data_;


//   void SetUp() override {
//     int send_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
//     int recv_data[3] = {};

//     InType test1 = std::make_tuple(
//         static_cast<const void*>(send_data1), // 1. sendbuf (const void*)
//         3,                                    // 2. sendcount (int)
//         MPI_INT,                              // 3. sendtype (MPI_Datatype)
//         static_cast<void*>(recv_data1),       // 4. recvbuf (void *)
//         3,                                    // 5. recvcount (int)
//         MPI_INT,                              // 6. recvtype (MPI_Datatype)
//         0,                                    // 7. root (int)
//         MPI_COMM_WORLD                        // 8. comm (MPI_Comm)
//     );


//   }

//   bool CheckTestOutputData(OutType &output_data) final {
//     const auto& input = GetTestInputData();
//     const auto& [sendbuf, sendcount, sendtype, recvbuf_dummy, recvcount, recvtype, root, comm] = input;

//     int rank = 0;
//     MPI_Comm_rank(comm, &rank);
    
//     if (recvcount > 0 && output_data == nullptr) {
//         return false;
//     }

//     int start_value = rank * recvcount + 1;
//     if (recvtype == MPI_INT) {
//         const int* actual_data = reinterpret_cast<const int*>(output_data);
//         for (int i = 0; i < recvcount; i++) {
//             if (actual_data[i] != start_value + i) {
//                 return false;
//             }
//             //std::cout << rank << "i:" << actual_data[i] << ' ';
//         }
//     } else if (recvtype == MPI_FLOAT) {
//         const float* actual_data = reinterpret_cast<const float*>(output_data);
//         for (int i = 0; i < recvcount; i++) {
//             if (std::abs(actual_data[i] - (float)(start_value + i)) >= 1e-6) {
//                 return false;
//             }
//             //std::cout << rank << "f:" << actual_data[i] << ' ';
//         }
//     } else if (recvtype == MPI_DOUBLE) {
//         const double* actual_data = reinterpret_cast<const double*>(output_data);
//         for (int i = 0; i < recvcount; i++) {
//             if (std::abs(actual_data[i] - (double)(start_value + i)) >= 1e-6) {
//                 return false;
//             }
//             //std::cout << rank << "d:" << actual_data[i] << ' ';
//         }
//     } else {
//         return false;
//     }

//     return true;
//   }

//   InType GetTestInputData() final {
//     return input_data_;
//   }
// };

// TEST_P(BaldinAMyScatterPerfTests, RunPerfModes) {
//   ExecuteTest(GetParam());
// }

// const auto kAllPerfTasks =
//     ppc::util::MakeAllPerfTasks<InType, BaldinAMyScatterMPI, BaldinAMyScatterSEQ>(PPC_SETTINGS_baldin_a_my_scatter);

// const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

// const auto kPerfTestName = BaldinAMyScatterPerfTests::CustomPerfTestName;

// INSTANTIATE_TEST_SUITE_P(RunModeTests, BaldinAMyScatterPerfTests, kGtestValues, kPerfTestName);

// }  // namespace baldin_a_my_scatter
