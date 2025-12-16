#include <gtest/gtest.h>
#include <mpi.h>
#include <numeric>
#include <vector>
#include <random>

#include "../../common/include/common.hpp"
#include "../../seq/include/reduce_seq.hpp"
#include "../../mpi/include/reduce_mpi.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"


namespace kutergin_v_reduce
{

class ReducePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType>
{
protected:
    // подготовка входных данных перед каждым тестом
    void SetUp() override
    {
        int rank = 0;
        if (ppc::util::IsUnderMpirun())
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        }

        std::mt19937 gen(rank); // rank выступает как seed для воспроизводимости
        std::uniform_int_distribution<> dis(1, 10);
        std::vector<int> local_vec(2000000);
        for (size_t i = 0; i < local_vec.size(); ++i)
        {
            local_vec[i] = dis(gen);
        }

        input_data_ = InType{local_vec, 0};
    }


    // проверка результата после выполнения задачи
    bool CheckTestOutputData(OutType& output_data) final
    {
        const auto& test_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());
        const auto& input = GetTestInputData(); // получение структуры {data, root}

        if (test_name.find("_seq_") != std::string::npos) 
        {
            const auto& input_vec = input.data;
            OutType expected_sum = std::accumulate(input_vec.begin(), input_vec.end(), 0);

            /*
            std::cout << "-> SEQ ACTUAL:   " << output_data << std::endl;
            std::cout << "-> SEQ EXPECTED: " << expected_sum << std::endl;
            */

            return output_data == expected_sum;
        }

        if (test_name.find("_mpi_") != std::string::npos) 
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            // корневой процесс возвращает результат проверки
            if (rank == input.root) 
            {
                // построение буфера для сбора
                int world_size;
                MPI_Comm_size(MPI_COMM_WORLD, &world_size);

                int expected_sum = 0;
                for (int r = 0; r < world_size; ++r) 
                {
                    std::mt19937 gen(r);
                    std::uniform_int_distribution<> dis(1, 10);
                    if (!input.data.empty())
                    {
                        expected_sum += dis(gen);
                    }
                }

                /*
                std::cout << "-> MPI ROOT ACTUAL:   " << output_data << std::endl;
                std::cout << "-> MPI ROOT EXPECTED: " << expected_sum << std::endl;
                */

                return output_data == expected_sum;
            }

            // рабочие процессы возвращают true
            return true;
        }
        return false;
    }

    
    // возврат подготовленных входных данных для задачи
    InType GetTestInputData() final
    {
        return input_data_;
    }

private:
    InType input_data_{};
};

namespace
{
    
TEST_P(ReducePerfTests, PerfTest)
{
    ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ReduceMPI, ReduceSequential>(PPC_SETTINGS_kutergin_v_reduce);

INSTANTIATE_TEST_SUITE_P(ReducePerf, ReducePerfTests, ppc::util::TupleToGTestValues(kAllPerfTasks), ReducePerfTests::CustomPerfTestName);

}

}