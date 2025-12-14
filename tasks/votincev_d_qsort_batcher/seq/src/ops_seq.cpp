#include "votincev_d_qsort_batcher/seq/include/ops_seq.hpp"

#include <algorithm>  // std::swap
#include <vector>

namespace votincev_d_qsort_batcher {

// нашел в интернете
void VotincevDQsortBatcherSEQ::QuickSort(double *arr, int left, int right) {
  // Вспомогательный стек для хранения границ подмассивов
  // Размер стека: максимальная глубина рекурсии (log N),
  // но для простоты возьмем максимальный размер массива
  // Выделяем на куче, чтобы избежать переполнения стека при больших N
  std::vector<int> stack(right - left + 1);

  // Инициализируем вершину стека
  int top = -1;

  // Вставляем начальные значения левой и правой границ в стек
  stack[++top] = left;
  stack[++top] = right;

  // Продолжаем, пока стек не опустеет
  while (top >= 0) {
    // Извлекаем правую (h) и левую (l) границы
    int h = stack[top--];
    int l = stack[top--];

    // Итеративный вызов функции разделения (partition)

    // ---------- Функция разделения (встроенная) ----------
    int i = l;
    int j = h;

    // Берем опорный элемент из середины
    double pivot = arr[(l + h) / 2];

    // Разделяем массив на две части
    while (i <= j) {
      while (arr[i] < pivot) {
        i++;
      }
      while (arr[j] > pivot) {
        j--;
      }

      if (i <= j) {
        std::swap(arr[i], arr[j]);
        i++;
        j--;
      }
    }

    // Индекс p - это граница разделения (i)
    int p = i;
    // Индексы разделения: [l, j] и [i, h]

    // Если есть элементы на левой стороне,
    // помещаем левый подмассив в стек
    if (l < j) {
      stack[++top] = l;
      stack[++top] = j;
    }

    // Если есть элементы на правой стороне,
    // помещаем правый подмассив в стек
    if (p < h) {
      stack[++top] = p;
      stack[++top] = h;
    }
  }
}

VotincevDQsortBatcherSEQ::VotincevDQsortBatcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

// входные данные — просто std::vector<double>, проверяем, что не пустой
bool VotincevDQsortBatcherSEQ::ValidationImpl() {
  const auto &vec = GetInput();
  return !vec.empty();
}

bool VotincevDQsortBatcherSEQ::PreProcessingImpl() {
  return true;
}

bool VotincevDQsortBatcherSEQ::RunImpl() {
  std::vector<double> data = GetInput();

  if (!data.empty()) {
    QuickSort(data.data(), 0, static_cast<int>(data.size()) - 1);
  }

  GetOutput() = data;
  return true;
}

bool VotincevDQsortBatcherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_qsort_batcher
