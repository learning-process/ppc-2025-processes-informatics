#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace guseva_a_jarvis {

using InType = std::tuple<int, int, std::vector<int>>;
using OutType = std::vector<int>;
using TestFromFileType = std::tuple<int, int, std::vector<int>, std::vector<int>>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

inline TestFromFileType ReadTestDataFromFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  int width = 0;
  int height = 0;
  std::vector<int> input;
  std::vector<int> output;
  file >> width >> height;
  std::string line;
  std::getline(file, line);
  std::getline(file, line);
  std::stringstream ss(line);
  int num = 0;
  while (ss >> num) {
    input.push_back(num);
  }
  std::getline(file, line);
  ss.clear();
  ss.str(line);
  while (ss >> num) {
    output.push_back(num);
  }
  file.close();
  return std::make_tuple(width, height, input, output);
}

}  // namespace guseva_a_jarvis
