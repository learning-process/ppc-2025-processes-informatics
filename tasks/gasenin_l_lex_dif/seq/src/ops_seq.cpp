#include "gasenin_l_lex_dif/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "gasenin_l_lex_dif/common/include/common.hpp"
#include "util/include/util.hpp"

namespace gasenin_l_lex_dif {

GaseninLLexDifSEQ::GaseninLLexDifSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GaseninLLexDifSEQ::ValidationImpl() {
  const auto &[str1, str2] = GetInput();
  return str1.length() <= 10000 && str2.length() <= 10000;
}

bool GaseninLLexDifSEQ::PreProcessingImpl() {
  return true;
}

bool GaseninLLexDifSEQ::RunImpl() {
  const auto &[str1, str2] = GetInput();

  size_t min_len = std::min(str1.length(), str2.length());

  for (size_t i = 0; i < min_len; ++i) {
    if (str1[i] != str2[i]) {
      GetOutput() = (str1[i] < str2[i]) ? -1 : 1;
      return true;
    }
  }

  if (str1.length() < str2.length()) {
    GetOutput() = -1;
  } else if (str1.length() > str2.length()) {
    GetOutput() = 1;
  } else {
    GetOutput() = 0;
  }

  return true;
}

bool GaseninLLexDifSEQ::PostProcessingImpl() {
  return true;
}

InType GaseninLLexDifSEQ::ReadInteractive() {
  InType input;

  std::cout << "=== Лексикографическое сравнение строк ===\n";
  std::cout << "Введите первую строку (макс. 10000 символов): ";
  if (!std::getline(std::cin, input.first)) {
    throw std::runtime_error("Ошибка чтения первой строки");
  }

  std::cout << "Введите вторую строку (макс. 10000 символов): ";
  if (!std::getline(std::cin, input.second)) {
    throw std::runtime_error("Ошибка чтения второй строки");
  }

  if (input.first.length() > 10000) {
    input.first = input.first.substr(0, 10000);
    std::cout << "Предупреждение: первая строка обрезана до 10000 символов\n";
  }

  if (input.second.length() > 10000) {
    input.second = input.second.substr(0, 10000);
    std::cout << "Предупреждение: вторая строка обрезана до 10000 символов\n";
  }

  return input;
}

InType GaseninLLexDifSEQ::ReadFromFile(const std::string &filename) {
  InType input;
  std::ifstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("Не удалось открыть файл: " + filename);
  }

  if (!std::getline(file, input.first)) {
    throw std::runtime_error("Не удалось прочитать первую строку из файла");
  }

  if (!std::getline(file, input.second)) {
    throw std::runtime_error("Не удалось прочитать вторую строку из файла");
  }

  file.close();

  if (input.first.length() > 10000) {
    input.first = input.first.substr(0, 10000);
  }

  if (input.second.length() > 10000) {
    input.second = input.second.substr(0, 10000);
  }

  return input;
}

void GaseninLLexDifSEQ::PrintResult(const InType &input, OutType result) {
  std::cout << "\n=== Результат сравнения ===\n";
  std::cout << "Первая строка: \"" << input.first << "\"\n";
  std::cout << "Вторая строка: \"" << input.second << "\"\n";

  switch (result) {
    case -1:
      std::cout << "Результат: Первая строка ЛЕКСИКОГРАФИЧЕСКИ МЕНЬШЕ второй\n";
      break;
    case 0:
      std::cout << "Результат: Строки ЛЕКСИКОГРАФИЧЕСКИ РАВНЫ\n";
      break;
    case 1:
      std::cout << "Результат: Первая строка ЛЕКСИКОГРАФИЧЕСКИ БОЛЬШЕ второй\n";
      break;
    default:
      std::cout << "Результат: Неизвестный результат\n";
  }

  std::cout << "\n--- Дополнительная информация ---\n";
  std::cout << "Длина первой строки: " << input.first.length() << " символов\n";
  std::cout << "Длина второй строки: " << input.second.length() << " символов\n";

  if (result != 0) {
    size_t min_len = std::min(input.first.length(), input.second.length());
    size_t diff_pos = min_len;

    for (size_t i = 0; i < min_len; ++i) {
      if (input.first[i] != input.second[i]) {
        diff_pos = i;
        break;
      }
    }

    if (diff_pos < min_len) {
      std::cout << "Первое различие на позиции: " << diff_pos << "\n";
      std::cout << "Символ в первой строке: '";
      if (std::isprint(input.first[diff_pos])) {
        std::cout << input.first[diff_pos];
      } else {
        std::cout << "\\x" << std::hex << static_cast<int>(input.first[diff_pos]);
      }
      std::cout << "' (код: " << std::dec << static_cast<int>(input.first[diff_pos]) << ")\n";

      std::cout << "Символ во второй строке: '";
      if (std::isprint(input.second[diff_pos])) {
        std::cout << input.second[diff_pos];
      } else {
        std::cout << "\\x" << std::hex << static_cast<int>(input.second[diff_pos]);
      }
      std::cout << "' (код: " << std::dec << static_cast<int>(input.second[diff_pos]) << ")\n";
    } else if (input.first.length() != input.second.length()) {
      std::cout << "Различие из-за разной длины строк\n";
      std::cout << "Более длинная строка: \"";
      if (input.first.length() > input.second.length()) {
        std::cout << input.first.substr(min_len);
      } else {
        std::cout << input.second.substr(min_len);
      }
      std::cout << "\"\n";
    }
  }
}

}  // namespace gasenin_l_lex_dif

#ifdef GASENIN_STANDALONE
#  include <iostream>

int main(int argc, char *argv[]) {
  try {
    gasenin_l_lex_dif::InType input;

    if (argc > 1) {
      std::string arg = argv[1];
      if (arg == "--file" || arg == "-f") {
        if (argc > 2) {
          input = gasenin_l_lex_dif::GaseninLLexDifSEQ::ReadFromFile(argv[2]);
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
      input = gasenin_l_lex_dif::GaseninLLexDifSEQ::ReadInteractive();
    }

    gasenin_l_lex_dif::GaseninLLexDifSEQ task(input);

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

    gasenin_l_lex_dif::GaseninLLexDifSEQ::PrintResult(input, task.GetOutput());

  } catch (const std::exception &e) {
    std::cerr << "Ошибка: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
#endif
