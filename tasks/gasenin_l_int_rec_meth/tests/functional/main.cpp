#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gasenin_l_int_rec_meth/common/include/common.hpp"
#include "gasenin_l_int_rec_meth/mpi/include/ops_mpi.hpp"
#include "gasenin_l_int_rec_meth/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gasenin_l_int_rec_meth {
/*
// Функция для standalone режима (оставляем, но не вызываем из main)
int RunStandalone(int argc, char* argv[]) {
  try {
    InType input;

    // Определяем режим работы
    if (argc > 1) {
      std::string arg = argv[1];
      if (arg == "--file" || arg == "-f") {
        if (argc > 2) {
          input = GaseninLIntRecMethSEQ::ReadFromFile(argv[2]);
        } else {
          std::cerr << "Ошибка: не указано имя файла для --file\n";
          return 1;
        }
      } else if (arg == "--help" || arg == "-h") {
        std::cout << "Использование:\n"
                  << "  " << argv[0] << "                    - интерактивный режим\n"
                  << "  " << argv[0] << " --file <filename>  - чтение из файла\n"
                  << "  " << argv[0] << " --help             - эта справка\n";
        return 0;
      } else {
        std::cerr << "Неизвестный аргумент: " << arg << "\n";
        std::cerr << "Используйте --help для справки\n";
        return 1;
      }
    } else {
      // Интерактивный режим
      input = GaseninLIntRecMethSEQ::ReadInteractive();
    }

    // Создаем и выполняем задачу
    GaseninLIntRecMethSEQ task(input);

    if (!task.Validation()) {
      std::cerr << "Ошибка: невалидные входные данные\n";
      return 1;
    }

    if (!task.PreProcessing()) {
      std::cerr << "Ошибка на этапе предобработки\n";
      return 1;
    }

    if (!task.Run()) {
      std::cerr << "Ошибка на этапе выполнения\n";
      return 1;
    }

    if (!task.PostProcessing()) {
      std::cerr << "Ошибка на этапе постобработки\n";
      return 1;
    }

    // Выводим результат
    GaseninLIntRecMethSEQ::PrintResult(input, task.GetOutput());

  } catch (const std::exception& e) {
    std::cerr << "Ошибка: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
*/
// Оригинальные тесты
class GaseninLRunFuncTestsIntRecMeth : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string test_name = std::get<1>(params);

    if (test_name == "apple_banana") {
      input_data_ = {"apple", "banana"};
      expected_output_ = -1;
    } else if (test_name == "hello_hello") {
      input_data_ = {"hello", "hello"};
      expected_output_ = 0;
    } else if (test_name == "zebra_apple") {
      input_data_ = {"zebra", "apple"};
      expected_output_ = 1;
    } else if (test_name == "empty_first") {
      input_data_ = {"", "test"};
      expected_output_ = -1;
    } else if (test_name == "empty_second") {
      input_data_ = {"test", ""};
      expected_output_ = 1;
    } else if (test_name == "both_empty") {
      input_data_ = {"", ""};
      expected_output_ = 0;
    } else if (test_name == "different_length") {
      input_data_ = {"abc", "abcd"};
      expected_output_ = -1;
    } else if (test_name == "same_prefix") {
      input_data_ = {"abcdef", "abcxyz"};
      expected_output_ = -1;
    } else if (test_name == "unicode_test") {
      input_data_ = {"привет", "пока"};
      expected_output_ = 1;  // 'р' > 'о' в Unicode
    } else if (test_name == "case_sensitive") {
      input_data_ = {"Apple", "apple"};
      expected_output_ = -1;  // 'A' < 'a' в ASCII
    } else {
      input_data_ = {"test", "test"};
      expected_output_ = 0;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_ = 0;
};

namespace {

TEST_P(GaseninLRunFuncTestsIntRecMeth, LexicographicComparison) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(1, "apple_banana"),     std::make_tuple(2, "hello_hello"),  std::make_tuple(3, "zebra_apple"),
    std::make_tuple(4, "empty_first"),      std::make_tuple(5, "empty_second"), std::make_tuple(6, "both_empty"),
    std::make_tuple(7, "different_length"), std::make_tuple(8, "same_prefix"),  std::make_tuple(9, "unicode_test"),
    std::make_tuple(10, "case_sensitive")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GaseninLIntRecMethMPI, InType>(kTestParam, PPC_SETTINGS_gasenin_l_int_rec_meth),
    ppc::util::AddFuncTask<GaseninLIntRecMethSEQ, InType>(kTestParam, PPC_SETTINGS_gasenin_l_int_rec_meth));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GaseninLRunFuncTestsIntRecMeth::PrintFuncTestName<GaseninLRunFuncTestsIntRecMeth>;

INSTANTIATE_TEST_SUITE_P(IntRecMethTests, GaseninLRunFuncTestsIntRecMeth, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gasenin_l_int_rec_meth
