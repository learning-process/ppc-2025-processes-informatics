#include "egorova_l_gauss_filter_vert/seq/include/ops_seq.hpp"
#include <algorithm>

namespace egorova_l_gauss_filter_vert {

EgorovaLGaussFilterVertSEQ::EgorovaLGaussFilterVertSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;  // Используем GetInput() для доступа к input_
}

bool EgorovaLGaussFilterVertSEQ::ValidationImpl() {
  auto& input = GetInput();  // Получаем ссылку на входные данные
  return input.rows >= 3 && input.cols >= 3;
}

bool EgorovaLGaussFilterVertSEQ::PreProcessingImpl() { return true; }

bool EgorovaLGaussFilterVertSEQ::RunImpl() {
  auto& in = GetInput();  // Получаем ссылку на входные данные
  auto& out = GetOutput(); // Получаем ссылку на выходные данные
  
  std::vector<uint8_t> result_data(in.data.size());

  const float ker[3][3] = {
      {0.0625f, 0.125f, 0.0625f}, 
      {0.125f, 0.25f, 0.125f}, 
      {0.0625f, 0.125f, 0.0625f}
  };

  for (int y = 0; y < in.rows; ++y) {
    for (int x = 0; x < in.cols; ++x) {
      for (int c = 0; c < in.channels; ++c) {
        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            int py = std::clamp(y + ky, 0, in.rows - 1);
            int px = std::clamp(x + kx, 0, in.cols - 1);
            sum += in.data[(py * in.cols + px) * in.channels + c] * ker[ky + 1][kx + 1];
          }
        }
        result_data[(y * in.cols + x) * in.channels + c] = static_cast<uint8_t>(sum);
      }
    }
  }
  
  // Сохраняем результат в output
  out.rows = in.rows;
  out.cols = in.cols;
  out.channels = in.channels;
  out.data = std::move(result_data);
  
  return true;
}

bool EgorovaLGaussFilterVertSEQ::PostProcessingImpl() { return true; }
}