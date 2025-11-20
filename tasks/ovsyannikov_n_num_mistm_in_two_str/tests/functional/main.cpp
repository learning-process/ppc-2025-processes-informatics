#include <gtest/gtest.h>

#include <array>
#include <string>
#include <utility>

#include "ovsyannikov_n_num_mistm_in_two_str/common/include/common.hpp"
#include "ovsyannikov_n_num_mistm_in_two_str/mpi/include/ops_mpi.hpp"
#include "ovsyannikov_n_num_mistm_in_two_str/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace ovsyannikov_n_num_mistm_in_two_str {

class OvsyannikovNRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, int> {
 protected:
  void SetUp() override {
    int elem_count = std::get<2>(GetParam());
    std::string seq_a(elem_count, 'a');
    std::string seq_b(elem_count, 'a');
    ref_val_ = 0;

    if (elem_count > 0) {
        // Несовпадения в первой половине строки
        int limit = elem_count / 2;
        for(int i = 0; i < limit; ++i) {
            seq_b[i] = 'b';
            ref_val_++;
        }
        // Несовпадение в конце
        if (elem_count > 5) {
            seq_b[elem_count - 1] = 'c';
            ref_val_++;
        }
    }
    
    task_data_ = std::make_pair(seq_a, seq_b);
  }

  bool CheckTestOutputData(OutType &actual_res) final {
    return actual_res == ref_val_;
  }

  InType GetTestInputData() final {
    return task_data_;
  }

 private:
  InType task_data_;
  int ref_val_ = 0;
};

TEST_P(OvsyannikovNRunFuncTestsProcesses, CalculateMismatches) {
  ExecuteTest(GetParam());
}

// Расширенный набор тестов
const std::array<int, 14> kTestLengths = {
    0,          // Пустая строка
    1,          // Минимальная строка
    3,          // Нечетное, малое
    17,         // Простое число
    64,         // Степень двойки 
    100,        // Стандартное
    127,        // Простое число около степени двойки
    1024,       
    54321,      
    100000,     
    1000000,    
    1234567,    
    10000000,   
    50000000    
};

const auto kTaskBundle =
    std::tuple_cat(ppc::util::AddFuncTask<OvsyannikovNNumMistmInTwoStrMPI, InType>(kTestLengths, PPC_SETTINGS_ovsyannikov_n_num_mistm_in_two_str),
                   ppc::util::AddFuncTask<OvsyannikovNNumMistmInTwoStrSEQ, InType>(kTestLengths, PPC_SETTINGS_ovsyannikov_n_num_mistm_in_two_str));

INSTANTIATE_TEST_SUITE_P(AllTests, OvsyannikovNRunFuncTestsProcesses, 
                         ppc::util::ExpandToValues(kTaskBundle));

}  // namespace ovsyannikov_n_num_mistm_in_two_str