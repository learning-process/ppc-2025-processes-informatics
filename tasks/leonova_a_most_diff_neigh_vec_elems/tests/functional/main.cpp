// #include <gtest/gtest.h>
// #include <stb/stb_image.h>

// #include <algorithm>
// #include <array>
// #include <cstddef>
// #include <cstdint>
// #include <numeric>
// #include <stdexcept>
// #include <string>
// #include <tuple>
// #include <utility>
// #include <vector>

// #include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"
// #include "leonova_a_most_diff_neigh_vec_elems/mpi/include/ops_mpi.hpp"
// #include "leonova_a_most_diff_neigh_vec_elems/seq/include/ops_seq.hpp"
// #include "util/include/func_test_util.hpp"
// #include "util/include/util.hpp"

// namespace leonova_a_most_diff_neigh_vec_elems {

// class MostDiffNeighVecElemsRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
//  public:
//   static std::string PrintTestParam(const TestType &test_param) {//строка которая возвращает параметры теста
//     //return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
//     return "";
//   }

//  protected:
//   void SetUp() override {//дает данные для теста
//     // int width = -1;
//     // int height = -1;
//     // int channels = -1;
//     // std::vector<uint8_t> img;
//     // // Read image in RGB to ensure consistent channel count
//     // {
//     //   std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_leonova_a_most_diff_neigh_vec_elems,
//     "pic.jpg");
//     //   auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, STBI_rgb);
//     //   if (data == nullptr) {
//     //     throw std::runtime_error("Failed to load image: " + std::string(stbi_failure_reason()));
//     //   }
//     //   channels = STBI_rgb;
//     //   img = std::vector<uint8_t>(data, data + (static_cast<ptrdiff_t>(width * height * channels)));
//     //   stbi_image_free(data);
//     //   if (std::cmp_not_equal(width, height)) {
//     //     throw std::runtime_error("width != height: ");
//     //   }
//     // }

//     // TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
//     // input_data_ = width - height + std::min(std::accumulate(img.begin(), img.end(), 0), channels);
//   }

//   bool CheckTestOutputData(OutType &output_data) final {//проверяетвыход
//     //return (input_data_ == output_data);
//     return true;
//   }

//   InType GetTestInputData() final {
//     return input_data_;
//   }

//  private:
//   InType input_data_ = std::vector<int>();
// };

// namespace {

// TEST_P(MostDiffNeighVecElemsRunFuncTestsProcesses, VecElemsDiff) {
//   ExecuteTest(GetParam());
// }

// const std::array<TestType, 3> kTestParam = {};//{std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7,
// "7")};

// const auto kTestTasksList =
//     std::tuple_cat(ppc::util::AddFuncTask<LeonovaAMostDiffNeighVecElemsMPI, InType>(kTestParam,
//     PPC_SETTINGS_leonova_a_most_diff_neigh_vec_elems),
//                    ppc::util::AddFuncTask<LeonovaAMostDiffNeighVecElemsSEQ, InType>(kTestParam,
//                    PPC_SETTINGS_leonova_a_most_diff_neigh_vec_elems));

// const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

// const auto kPerfTestName =
// MostDiffNeighVecElemsRunFuncTestsProcesses::PrintFuncTestName<MostDiffNeighVecElemsRunFuncTestsProcesses>;

// INSTANTIATE_TEST_SUITE_P(VecTests, MostDiffNeighVecElemsRunFuncTestsProcesses, kGtestValues,
// kPerfTestName);//kPerfTestName

// }  // namespace

// }  // namespace leonova_a_most_diff_neigh_vec_elems

#include <gtest/gtest.h>

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

#include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"
#include "leonova_a_most_diff_neigh_vec_elems/mpi/include/ops_mpi.hpp"
#include "leonova_a_most_diff_neigh_vec_elems/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace leonova_a_most_diff_neigh_vec_elems {

class MostDiffNeighVecElemsRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  // static std::string PrintTestParam(const TestType &test_param) {
  //   // Возвращаем строковое представление тестового параметра
  //   auto expected_pair = std::get<0>(test_param);
  //   return "expected_" + std::to_string(std::get<0>(expected_pair)) + "_" +
  //          std::to_string(std::get<1>(expected_pair));
  // }
  // static std::string PrintTestParam(const TestType &test_param) {
  //   auto expected_pair = std::get<0>(test_param);
  //   // Используем только буквы, цифры и подчеркивания
  //   return "expected_" + std::to_string(std::get<0>(expected_pair)) + "_" +
  //          std::to_string(std::get<1>(expected_pair));
  // }
  static std::string PrintTestParam(const TestType &test_param) {
    auto expected_pair = std::get<0>(test_param);
    int first = std::get<0>(expected_pair);
    int second = std::get<1>(expected_pair);

    std::string first_str = (first < 0) ? "neg" + std::to_string(-first) : std::to_string(first);
    std::string second_str = (second < 0) ? "neg" + std::to_string(-second) : std::to_string(second);

    return "expected_" + first_str + "_" + second_str;
  }

 protected:
  void SetUp() override {
    // Получаем параметры теста
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    // input_data_ - это вектор для обработки
    input_data_ = std::get<1>(params);

    // expected_output_ - ожидаемая пара элементов
    expected_output_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем, что выходные данные соответствуют ожидаемым
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = std::vector<int>();
  OutType expected_output_ = std::tuple<int, int>(0, 0);
};

namespace {

TEST_P(MostDiffNeighVecElemsRunFuncTestsProcesses, VecElemsDiff) {
  ExecuteTest(GetParam());
}

// Определяем тестовые случаи
const std::array<TestType, 5> kTestParam = {
    std::make_tuple(std::make_tuple(1, 10), std::vector<int>{1, 10, 3, 10, 5}),   // Максимальная разница между 1 и 10
    std::make_tuple(std::make_tuple(5, 20), std::vector<int>{5, 20, 15, 10}),     // Максимальная разница между 5 и 20
    std::make_tuple(std::make_tuple(7, 7), std::vector<int>{7}),                  // Один элемент
    std::make_tuple(std::make_tuple(0, 100), std::vector<int>{0, 100, 100, 25}),  // Максимальная разница между 0 и 100
    std::make_tuple(std::make_tuple(-10, -5), std::vector<int>{-10, -5, -1, 3, 0, 0})};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LeonovaAMostDiffNeighVecElemsMPI, InType>(
                                               kTestParam, PPC_SETTINGS_leonova_a_most_diff_neigh_vec_elems),
                                           ppc::util::AddFuncTask<LeonovaAMostDiffNeighVecElemsSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_leonova_a_most_diff_neigh_vec_elems));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    MostDiffNeighVecElemsRunFuncTestsProcesses::PrintFuncTestName<MostDiffNeighVecElemsRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(VecTests, MostDiffNeighVecElemsRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace leonova_a_most_diff_neigh_vec_elems
