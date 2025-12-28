#pragma once

#include <tuple>
#include <vector>

#include "chaschin_v_sobel_operator/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chaschin_v_sobel_operator {

class ChaschinVSobelOperatorMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ChaschinVSobelOperatorMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<float> preprocess_to_grayscale(const std::vector<std::vector<Pixel>> &image);

      std::vector<float> PreProcessGray;
  std::vector<float> PostProcessGray;
  std::tuple<int, int> Size;
  std::vector<int> ScatterSendCounts;
  std::vector<int> ScatterDispls;
};

}  // namespace chaschin_v_sobel_operator
