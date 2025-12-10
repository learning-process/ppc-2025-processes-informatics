#include <gtest/gtest.h>
#include <vector>
#include <tuple>

#include <mpi.h>
#include "makovskiy_i_gauss_filter_vert/common/include/common.hpp"

#include "makovskiy_i_gauss_filter_vert/seq/include/ops_seq.hpp"
#include "makovskiy_i_gauss_filter_vert/mpi/include/ops_mpi.hpp"

using namespace makovskiy_i_gauss_filter_vert;

TEST(GaussFilter_MPI_Func_Test, Correctness_3x3_Image) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int width = 3;
    const int height = 3;
    std::vector<int> input_image = {
        10, 20, 30,
        40, 50, 60,
        70, 80, 90
    };
    
    std::vector<int> expected_output = {
        20, 27, 35,
        42, 50, 57,
        65, 72, 80
    };
    
    InType input_data = std::make_tuple(input_image, width, height);
    
    auto mpi_task = std::make_shared<GaussFilterMPI>(input_data);
    
    ASSERT_TRUE(mpi_task->Validation());
    ASSERT_TRUE(mpi_task->PreProcessing());
    ASSERT_TRUE(mpi_task->Run());
    ASSERT_TRUE(mpi_task->PostProcessing());

    if (rank == 0) {
        auto mpi_result = mpi_task->GetOutput();
        ASSERT_EQ(mpi_result, expected_output);
    }
}


TEST(GaussFilter_SEQ_Func_Test, Correctness_4x4_Image) {
    const int width = 4;
    const int height = 4;
    std::vector<int> input_image = {
        0,   10,  20,  30,
        40,  50,  60,  70,
        80,  90,  100, 110,
        120, 130, 140, 150
    };
    
    std::vector<int> expected_output = {
        12, 20, 30, 37,
        42, 50, 60, 67,
        82, 90, 100, 107,
        112, 120, 130, 137
    };
    
    InType input_data = std::make_tuple(input_image, width, height);

    auto seq_task = std::make_shared<GaussFilterSEQ>(input_data);
    
    ASSERT_TRUE(seq_task->Validation());
    ASSERT_TRUE(seq_task->PreProcessing());
    ASSERT_TRUE(seq_task->Run());
    ASSERT_TRUE(seq_task->PostProcessing());
    
    auto seq_result = seq_task->GetOutput();
    ASSERT_EQ(seq_result, expected_output);
}